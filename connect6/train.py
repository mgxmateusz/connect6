from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .checkpoint import CheckpointManager, load_checkpoint
from .config import load_config
from .logger import TrainingLogger
from .model import build_model, mask_logits
from .utils import resolve_device, seed_everything
from .vector_env import VectorConnect6, canonical_network_input


UNKNOWN_EPISODE_RESULT = 2  # poprawne wyniki: -1, 0, +1


class CompleteGameBuffer:
    """Bufor GPU dla terminalnych segmentów jednego update'u PPO.

    Plansza może wejść w update już w połowie partii rozegranej przez poprzednią
    wersję modelu. Do bufora trafiają tylko decyzje aktualnego modelu.

    Jeżeli partia skończy się podczas bieżącego update'u, cały segment od początku
    tego update'u dostaje prawdziwy wynik terminalny i może trafić do PPO.

    Jeżeli partia nie skończy się przed PPO, historia segmentu jest odrzucana,
    ale sama plansza NIE jest resetowana. Następny model przejmuje aktualny stan.
    """

    def __init__(
        self,
        target_completed_positions: int,
        envs: int,
        board_size: int,
        device: torch.device,
    ) -> None:

        self.target_completed_positions = int(target_completed_positions)
        self.envs = int(envs)
        self.board_size = int(board_size)

        self.max_game_positions = (
            self.board_size
            * self.board_size
        )

        # Target + maksymalnie jeden niedokończony segment
        # na każdy env + ostatni wektorowy krok.
        self.capacity = (
            self.target_completed_positions
            + self.envs * self.max_game_positions
            + self.envs
        )

        self.boards = torch.empty(
            (
                self.capacity,
                board_size,
                board_size,
            ),
            dtype=torch.int8,
            device=device,
        )

        self.players = torch.empty(
            self.capacity,
            dtype=torch.int8,
            device=device,
        )

        self.stones_left = torch.empty(
            self.capacity,
            dtype=torch.int8,
            device=device,
        )

        self.actions = torch.empty(
            self.capacity,
            dtype=torch.int16,
            device=device,
        )

        self.logprobs = torch.empty(
            self.capacity,
            dtype=torch.float32,
            device=device,
        )

        self.values = torch.empty(
            self.capacity,
            dtype=torch.float32,
            device=device,
        )

        self.episode_ids = torch.empty(
            self.capacity,
            dtype=torch.int32,
            device=device,
        )

        self.count = 0


    def reset(self) -> None:

        # Czyścimy wyłącznie historię treningową.
        # Pamięć pozostaje zaalokowana.
        self.count = 0


    def append_batch(
        self,
        boards: torch.Tensor,
        players: torch.Tensor,
        stones_left: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> None:

        batch = int(
            boards.shape[0]
        )

        start = self.count
        end = start + batch

        if end > self.capacity:

            raise RuntimeError(
                "Przepełnienie CompleteGameBuffer. "
                "Zwiększ completed_positions_per_update "
                "albo sprawdź logikę końca gry."
            )


        self.boards[
            start:end
        ].copy_(
            boards
        )

        self.players[
            start:end
        ].copy_(
            players
        )

        self.stones_left[
            start:end
        ].copy_(
            stones_left
        )

        self.actions[
            start:end
        ].copy_(
            actions.to(
                torch.int16
            )
        )

        self.logprobs[
            start:end
        ].copy_(
            logprobs
        )

        self.values[
            start:end
        ].copy_(
            values
        )

        self.episode_ids[
            start:end
        ].copy_(
            episode_ids
        )

        self.count = end


    def completed_samples(
        self,
        episode_results: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        """Zwraca próbki z terminalnych segmentów
        oraz wynik z perspektywy aktora.
        """

        used_episode_ids = (
            self.episode_ids[
                :self.count
            ]
            .long()
        )


        winners = (
            episode_results[
                used_episode_ids
            ]
        )


        complete_mask = (
            winners.ne(
                UNKNOWN_EPISODE_RESULT
            )
        )


        indices = torch.nonzero(
            complete_mask,
            as_tuple=False,
        ).flatten()


        # winner:
        # +1 = czarne
        # -1 = białe
        #  0 = remis
        #
        # player:
        # +1 = czarne
        # -1 = białe
        #
        # iloczyn:
        # +1 = aktor wygrał
        # -1 = aktor przegrał
        #  0 = remis

        actor_outcomes = (
            winners[
                indices
            ].to(
                torch.float32
            )

            *

            self.players[
                indices
            ].to(
                torch.float32
            )
        )


        return (
            indices,
            actor_outcomes,
        )



# =============================================================================
# AMP
# =============================================================================

def _autocast_context(
    device: torch.device,
    enabled: bool,
    dtype_name: str,
):

    if (
        not enabled
        or device.type != "cuda"
    ):

        return torch.autocast(
            device_type=device.type,
            enabled=False,
        )


    dtype = (
        torch.bfloat16
        if dtype_name.lower() == "bfloat16"
        else torch.float16
    )


    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=True,
    )



# =============================================================================
# TEMPERATURA
# =============================================================================

def _temperature(
    update: int,
    cfg: dict[str, Any],
) -> float:

    start = float(
        cfg.get(
            "temperature_start",
            1.0,
        )
    )

    end = float(
        cfg.get(
            "temperature_end",
            0.25,
        )
    )

    decay = max(
        1,
        int(
            cfg.get(
                "temperature_decay_updates",
                5000,
            )
        ),
    )


    progress = min(
        1.0,
        update / decay,
    )


    return (
        start
        + (
            end
            - start
        )
        * progress
    )



# =============================================================================
# LEARNING RATE
# =============================================================================

def _learning_rate(
    update: int,
    total_updates: int,
    base_lr: float,
    schedule: str,
) -> float:

    if schedule == "constant":

        return base_lr


    if schedule == "cosine":

        progress = min(
            1.0,
            update
            / max(
                1,
                total_updates,
            ),
        )


        return (
            base_lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )


    raise ValueError(
        f"Nieznany lr_schedule: {schedule}"
    )



# =============================================================================
# STATYSTYKI
# =============================================================================

def _mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )



def _percentile(
    values: list[float],
    percentile: float,
) -> float:

    """Percentyl z interpolacją liniową."""

    if not values:
        return 0.0


    ordered = sorted(
        values
    )


    if len(ordered) == 1:
        return ordered[0]


    position = (
        percentile
        * (
            len(ordered)
            - 1
        )
    )


    lower = int(
        math.floor(
            position
        )
    )


    upper = int(
        math.ceil(
            position
        )
    )


    if lower == upper:
        return ordered[
            lower
        ]


    fraction = (
        position
        - lower
    )


    return (
        ordered[
            lower
        ]
        +
        (
            ordered[
                upper
            ]
            -
            ordered[
                lower
            ]
        )
        * fraction
    )



# =============================================================================
# TRENING
# =============================================================================

def train(
    config_path: str | Path,
) -> None:

    cfg = load_config(
        config_path
    )


    run_cfg = cfg[
        "run"
    ]

    game_cfg = cfg[
        "game"
    ]

    model_cfg = cfg[
        "model"
    ]

    tr = cfg[
        "training"
    ]


    seed_everything(
        int(
            run_cfg.get(
                "seed",
                42,
            )
        )
    )


    device = resolve_device(
        str(
            tr.get(
                "device",
                "cuda",
            )
        )
    )


    if device.type == "cuda":

        torch.set_float32_matmul_precision(
            "high"
        )


    run_dir = (
        Path(
            run_cfg.get(
                "root_dir",
                "runs",
            )
        )
        /
        str(
            run_cfg.get(
                "name",
                "connect6",
            )
        )
    )


    checkpoint_mgr = CheckpointManager(
        run_dir
        / "checkpoints"
    )


    logger = TrainingLogger(
        run_dir
    )


    board_size = int(
        game_cfg.get(
            "board_size",
            19,
        )
    )


    win_length = int(
        game_cfg.get(
            "win_length",
            6,
        )
    )


    num_envs = int(
        tr.get(
            "num_envs",
            1024,
        )
    )


    completed_target = int(
        tr.get(
            "completed_positions_per_update",
            256,
        ) * tr.get(
            "minibatch_size",
            2048,
        )
    )


    total_updates = int(
        tr.get(
            "updates",
            100000,
        )
    )


    if completed_target <= 0:

        raise ValueError(
            "completed_positions_per_update "
            "musi być większe od 0"
        )


    # =========================================================================
    # MODEL
    # =========================================================================

    model = build_model(
        model_cfg,
        board_size,
    ).to(
        device
    )


    base_lr = float(
        tr.get(
            "learning_rate",
            3e-4,
        )
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=base_lr,

        weight_decay=float(
            tr.get(
                "weight_decay",
                1e-4,
            )
        ),

        eps=float(
            tr.get(
                "adam_eps",
                1e-5,
            )
        ),
    )


    # =========================================================================
    # AMP
    # =========================================================================

    use_amp = (
        bool(
            tr.get(
                "amp",
                True,
            )
        )
        and
        device.type == "cuda"
    )


    amp_dtype_name = str(
        tr.get(
            "amp_dtype",
            "bfloat16",
        )
    )


    scaler_enabled = (
        use_amp
        and
        amp_dtype_name.lower()
        == "float16"
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
    )


    # =========================================================================
    # CHECKPOINT / RESUME
    # =========================================================================

    start_update = 0
    global_step = 0


    resume_mode = str(
        run_cfg.get(
            "resume",
            "auto",
        )
    )


    resume_path: Path | None = None


    if resume_mode.lower() == "auto":

        resume_path = (
            checkpoint_mgr.find_latest()
        )


    elif resume_mode.lower() not in (
        "none",
        "off",
        "false",
        "",
    ):

        resume_path = Path(
            resume_mode
        )


    if (
        resume_path
        and
        resume_path.exists()
    ):

        payload = load_checkpoint(
            resume_path,
            map_location="cpu",
        )


        checkpoint_model_cfg = (
            payload.get(
                "model_config",
                {},
            )
        )


        checkpoint_version = int(
            checkpoint_model_cfg.get(
                "architecture_version",
                1,
            )
        )


        current_version = int(
            model_cfg.get(
                "architecture_version",
                3,
            )
        )


        if (
            checkpoint_version
            != current_version
        ):

            raise RuntimeError(
                f"Checkpoint {resume_path} "
                "używa innej architektury modelu. "
                "Ustaw nową nazwę runa "
                "albo run.resume: none."
            )


        model.load_state_dict(
            payload[
                "model_state"
            ]
        )


        optimizer.load_state_dict(
            payload[
                "optimizer_state"
            ]
        )


        if (
            payload.get(
                "scaler_state"
            )
            and
            scaler_enabled
        ):

            scaler.load_state_dict(
                payload[
                    "scaler_state"
                ]
            )


        start_update = (
            int(
                payload.get(
                    "update",
                    0,
                )
            )
            + 1
        )


        global_step = int(
            payload.get(
                "global_step",
                0,
            )
        )


        print(
            f"[resume] "
            f"{resume_path} "
            f"-> update "
            f"{start_update}, "
            f"global_step "
            f"{global_step}"
        )


    # =========================================================================
    # torch.compile
    # =========================================================================

    if bool(
        model_cfg.get(
            "compile",
            False,
        )
    ):

        compile_mode = str(
            model_cfg.get(
                "compile_mode",
                "default",
            )
        )


        print(
            f"[compile] "
            f"torch.compile("
            f"mode="
            f"{compile_mode!r}"
            f")"
        )


        model = torch.compile(
            model,
            mode=compile_mode,
        )


    # =========================================================================
    # ŚRODOWISKO
    # =========================================================================

    env = VectorConnect6(
        num_envs=num_envs,
        board_size=board_size,
        win_length=win_length,
        device=device,
        debug_checks=bool(
            game_cfg.get(
                "debug_checks",
                False,
            )
        ),
    )


    buffer = CompleteGameBuffer(
        completed_target,
        num_envs,
        board_size,
        device,
    )


    # =========================================================================
    # PPO CONFIG
    # =========================================================================

    clip_coef = float(
        tr.get(
            "clip_coef",
            0.2,
        )
    )


    value_coef = float(
        tr.get(
            "value_coef",
            0.5,
        )
    )


    entropy_coef = float(
        tr.get(
            "entropy_coef",
            0.01,
        )
    )


    max_grad_norm = float(
        tr.get(
            "max_grad_norm",
            1.0,
        )
    )


    ppo_epochs = int(
        tr.get(
            "ppo_epochs",
            4,
        )
    )


    minibatch_size = int(
        tr.get(
            "minibatch_size",
            2048,
        )
    )


    target_kl = float(
        tr.get(
            "target_kl",
            0.03,
        )
    )


    normalize_adv = bool(
        tr.get(
            "normalize_advantage",
            True,
        )
    )


    checkpoint_every = max(
        1,
        int(
            tr.get(
                "checkpoint_every_updates",
                25,
            )
        ),
    )


    dashboard_every = max(
        1,
        int(
            tr.get(
                "dashboard_every_updates",
                5,
            )
        ),
    )


    lr_schedule = str(
        tr.get(
            "lr_schedule",
            "cosine",
        )
    ).lower()


    # PPO wymaga, aby old_logprob
    # odpowiadał dokładnie temu samemu
    # stanowi i akcji.

    if bool(
        tr.get(
            "symmetry_augmentation",
            False,
        )
    ):

        print(
            "[uwaga] symmetry_augmentation "
            "jest ignorowane: "
            "obracanie próbki po jej zebraniu "
            "ze starym old_logprob "
            "zepsułoby ratio PPO."
        )


    games_total = 0


    # =========================================================================
    # INFORMACJE STARTOWE
    # =========================================================================

    print(
        f"[urządzenie] "
        f"{device}"
    )


    if device.type == "cuda":

        print(
            f"[gpu] "
            f"{torch.cuda.get_device_name(device)} "
            f"| CUDA runtime "
            f"{torch.version.cuda}"
        )


    print(
        f"[środowisko] "
        f"{num_envs} parallel boards "
        f"| terminal-segment target >= "
        f"{completed_target:,} "
        f"positions/update"
    )


    print(
        f"[bufor] "
        f"max temporary collection capacity "
        f"{buffer.capacity:,} positions "
        f"| unfinished histories "
        f"are discarded, "
        f"boards persist across PPO updates"
    )


    print(
        f"[model] parameters: "
        f"{sum(
            p.numel()
            for p
            in getattr(
                model,
                '_orig_mod',
                model,
            ).parameters()
        ):,}"
    )


    print(
        f"[run] "
        f"{run_dir.resolve()}"
    )


    # =========================================================================
    # GŁÓWNA PĘTLA
    # =========================================================================

    try:

        for update in range(
            start_update,
            total_updates,
        ):

            update_started = (
                time.perf_counter()
            )


            temperature = (
                _temperature(
                    update,
                    tr,
                )
            )


            lr = _learning_rate(
                update,
                total_updates,
                base_lr,
                lr_schedule,
            )


            for group in (
                optimizer.param_groups
            ):

                group[
                    "lr"
                ] = lr


            # =================================================================
            # 1) SELF-PLAY
            # =================================================================
            #
            # UWAGA:
            # NIE ROBIMY env.reset().
            #
            # Plansze z poprzedniego PPO zostają.
            # Czyścimy tylko historię poprzedniej
            # wersji modelu.
            # =================================================================

            buffer.reset()

            model.eval()


            update_games = 0
            update_black = 0
            update_white = 0
            update_draws = 0

            completed_positions = 0

            game_lengths: list[
                float
            ] = []


            # Ile decyzji AKTUALNY model
            # wykonał w obecnym segmencie
            # na każdym stole.

            current_segment_lengths = (
                torch.zeros(
                    num_envs,
                    dtype=torch.int32,
                    device=device,
                )
            )


            # ID dotyczą segmentów
            # tego update'u.
            #
            # Nie są to globalne ID
            # całej historii gry.

            current_episode_ids = (
                torch.arange(
                    num_envs,
                    dtype=torch.int32,
                    device=device,
                )
            )


            next_episode_id = (
                num_envs
            )


            episode_results = torch.full(
                (
                    buffer.capacity
                    + num_envs
                    + 1,
                ),
                UNKNOWN_EPISODE_RESULT,
                dtype=torch.int8,
                device=device,
            )


            collection_started = (
                time.perf_counter()
            )


            with torch.inference_mode():

                while (
                    completed_positions
                    < completed_target
                ):

                    # ---------------------------------------------------------
                    # STAN
                    # ---------------------------------------------------------

                    network_input = (
                        env.network_input()
                    )


                    legal = (
                        env.legal_mask()
                    )


                    # ---------------------------------------------------------
                    # MODEL
                    # ---------------------------------------------------------

                    with _autocast_context(
                        device,
                        use_amp,
                        amp_dtype_name,
                    ):

                        logits, values = (
                            model(
                                network_input
                            )
                        )


                    logits = (
                        mask_logits(
                            logits.float(),
                            legal,
                        )
                        /
                        max(
                            temperature,
                            1e-4,
                        )
                    )


                    dist = Categorical(
                        logits=logits
                    )


                    actions = (
                        dist.sample()
                    )


                    old_logprobs = (
                        dist.log_prob(
                            actions
                        )
                    )


                    # ---------------------------------------------------------
                    # ZAPIS STANU PRZED RUCHEM
                    # ---------------------------------------------------------

                    buffer.append_batch(
                        boards=env.boards,
                        players=env.current_player,
                        stones_left=env.stones_left,
                        actions=actions,
                        logprobs=old_logprobs,
                        values=values.float(),
                        episode_ids=current_episode_ids,
                    )


                    # ---------------------------------------------------------
                    # RUCH
                    # ---------------------------------------------------------

                    step = env.step(
                        actions
                    )


                    # Każdy aktywny env wykonał
                    # dokładnie jedną decyzję
                    # aktualnego modelu.

                    current_segment_lengths += 1


                    done_idx = torch.nonzero(
                        step.done,
                        as_tuple=False,
                    ).flatten()


                    # ---------------------------------------------------------
                    # KONIEC GRY
                    # ---------------------------------------------------------

                    if done_idx.numel():

                        done_episode_ids = (
                            current_episode_ids[
                                done_idx
                            ]
                            .long()
                        )


                        winners = (
                            step.winner[
                                done_idx
                            ]
                        )


                        # Pełna długość partii:
                        # tylko statystyka.

                        full_game_lengths = (
                            step.game_lengths[
                                done_idx
                            ]
                            .long()
                        )


                        # Długość fragmentu
                        # należącego do AKTUALNEGO
                        # modelu.

                        segment_lengths = (
                            current_segment_lengths[
                                done_idx
                            ]
                            .long()
                        )


                        # Wynik terminalny
                        # dla segmentu.

                        episode_results[
                            done_episode_ids
                        ] = winners


                        completed_positions += int(
                            segment_lengths
                            .sum()
                            .item()
                        )


                        # -----------------------------------------------------
                        # STATYSTYKI
                        # -----------------------------------------------------

                        update_games += int(
                            done_idx.numel()
                        )


                        update_black += int(
                            winners
                            .eq(1)
                            .sum()
                            .item()
                        )


                        update_white += int(
                            winners
                            .eq(-1)
                            .sum()
                            .item()
                        )


                        update_draws += int(
                            winners
                            .eq(0)
                            .sum()
                            .item()
                        )


                        game_lengths.extend(
                            full_game_lengths
                            .float()
                            .cpu()
                            .tolist()
                        )


                        # -----------------------------------------------------
                        # RESET TYLKO ZAKOŃCZONYCH GIER
                        # -----------------------------------------------------

                        env.reset(
                            done_idx
                        )


                        current_segment_lengths[
                            done_idx
                        ] = 0


                        # Nowa partia
                        # = nowy segment.

                        count_done = int(
                            done_idx.numel()
                        )


                        new_ids = torch.arange(
                            next_episode_id,
                            next_episode_id
                            + count_done,
                            dtype=torch.int32,
                            device=device,
                        )


                        current_episode_ids[
                            done_idx
                        ] = new_ids


                        next_episode_id += (
                            count_done
                        )


            if device.type == "cuda":

                torch.cuda.synchronize(
                    device
                )


            collection_elapsed = max(
                1e-9,
                time.perf_counter()
                - collection_started,
            )


            # =================================================================
            # WYBÓR TYLKO TERMINALNYCH SEGMENTÓW
            # =================================================================

            completed_idx, returns = (
                buffer.completed_samples(
                    episode_results
                )
            )


            train_size = int(
                completed_idx.numel()
            )


            generated_positions = int(
                buffer.count
            )


            discarded_positions = (
                generated_positions
                - train_size
            )


            # Kontrola poprawności.
            #
            # train_size musi być równy
            # sumie długości segmentów,
            # które faktycznie zakończyły grę.

            if (
                train_size
                != completed_positions
            ):

                raise RuntimeError(
                    "Collector accounting mismatch: "
                    f"terminal segment positions="
                    f"{completed_positions}, "
                    f"completed samples="
                    f"{train_size}."
                )


            # Liczba niedokończonych
            # segmentów, których HISTORIĘ
            # wyrzucamy.
            #
            # PLANSZ NIE RESETUJEMY.

            unfinished_games_discarded = int(
                current_segment_lengths
                .gt(0)
                .sum()
                .item()
            )


            # =================================================================
            # ADVANTAGE
            # =================================================================

            advantages = (
                returns
                -
                buffer.values[
                    completed_idx
                ]
            )


            if normalize_adv:

                advantages = (
                    advantages
                    -
                    advantages.mean()
                ) / (
                    advantages.std(
                        unbiased=False
                    )
                    + 1e-8
                )


            # =================================================================
            # 2) PPO
            # =================================================================

            model.train()


            losses: list[
                float
            ] = []

            policy_losses: list[
                float
            ] = []

            value_losses: list[
                float
            ] = []

            entropies: list[
                float
            ] = []

            kls: list[
                float
            ] = []

            clipfracs: list[
                float
            ] = []


            # Gradienty mierzone
            # PRZED clippingiem.

            grad_norms: list[
                float
            ] = []

            grad_clip_scales: list[
                float
            ] = []

            grad_was_clipped: list[
                float
            ] = []


            stop_for_kl = False


            for _epoch in range(
                ppo_epochs
            ):

                order = torch.randperm(
                    train_size,
                    device=device,
                )


                for start in range(
                    0,
                    train_size,
                    minibatch_size,
                ):

                    local_idx = (
                        order[
                            start:
                            start
                            + minibatch_size
                        ]
                    )


                    idx = (
                        completed_idx[
                            local_idx
                        ]
                    )


                    # ---------------------------------------------------------
                    # MINIBATCH
                    # ---------------------------------------------------------

                    mb_boards = (
                        buffer.boards[
                            idx
                        ]
                    )


                    mb_players = (
                        buffer.players[
                            idx
                        ]
                    )


                    mb_stones = (
                        buffer.stones_left[
                            idx
                        ]
                    )


                    mb_actions = (
                        buffer.actions[
                            idx
                        ]
                        .long()
                    )


                    mb_input = (
                        canonical_network_input(
                            mb_boards,
                            mb_players,
                            mb_stones,
                        )
                    )


                    mb_legal = (
                        mb_boards
                        .reshape(
                            mb_boards.shape[0],
                            -1,
                        )
                        .eq(0)
                    )


                    # ---------------------------------------------------------
                    # FORWARD
                    # ---------------------------------------------------------

                    with _autocast_context(
                        device,
                        use_amp,
                        amp_dtype_name,
                    ):

                        new_logits, new_values = (
                            model(
                                mb_input
                            )
                        )


                        new_logits = (
                            mask_logits(
                                new_logits,
                                mb_legal,
                            )
                            /
                            max(
                                temperature,
                                1e-4,
                            )
                        )


                        dist = Categorical(
                            logits=
                            new_logits.float()
                        )


                        new_logprob = (
                            dist.log_prob(
                                mb_actions
                            )
                        )


                        entropy = (
                            dist.entropy()
                            .mean()
                        )


                        # -----------------------------------------------------
                        # PPO RATIO
                        # -----------------------------------------------------

                        old_logprob = (
                            buffer.logprobs[
                                idx
                            ]
                        )


                        logratio = (
                            new_logprob
                            -
                            old_logprob
                        )


                        ratio = (
                            logratio.exp()
                        )


                        mb_adv = (
                            advantages[
                                local_idx
                            ]
                        )


                        pg1 = (
                            -mb_adv
                            * ratio
                        )


                        pg2 = (
                            -mb_adv
                            *
                            torch.clamp(
                                ratio,
                                1.0
                                - clip_coef,
                                1.0
                                + clip_coef,
                            )
                        )


                        policy_loss = (
                            torch.maximum(
                                pg1,
                                pg2,
                            )
                            .mean()
                        )


                        # -----------------------------------------------------
                        # VALUE
                        # -----------------------------------------------------

                        value_pred = (
                            new_values.float()
                        )


                        value_loss = (
                            0.5
                            *
                            (
                                value_pred
                                -
                                returns[
                                    local_idx
                                ]
                            )
                            .pow(2)
                            .mean()
                        )


                        # -----------------------------------------------------
                        # TOTAL LOSS
                        # -----------------------------------------------------

                        loss = (
                            policy_loss
                            +
                            value_coef
                            * value_loss
                            -
                            entropy_coef
                            * entropy
                        )


                    # ---------------------------------------------------------
                    # BACKPROP
                    # ---------------------------------------------------------

                    optimizer.zero_grad(
                        set_to_none=True
                    )


                    if scaler_enabled:

                        scaler.scale(
                            loss
                        ).backward()


                        scaler.unscale_(
                            optimizer
                        )


                    else:

                        loss.backward()


                    # ---------------------------------------------------------
                    # GRADIENT CLIPPING
                    # ---------------------------------------------------------
                    #
                    # clip_grad_norm_ ZWRACA
                    # normę SPRZED clippingu.
                    # ---------------------------------------------------------

                    grad_norm_tensor = (
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    )


                    grad_norm = float(
                        grad_norm_tensor
                        .detach()
                        .item()
                    )


                    grad_scale = min(
                        1.0,

                        max_grad_norm
                        /
                        (
                            grad_norm
                            + 1e-6
                        ),
                    )


                    grad_norms.append(
                        grad_norm
                    )


                    grad_clip_scales.append(
                        grad_scale
                    )


                    grad_was_clipped.append(
                        1.0
                        if (
                            grad_norm
                            >
                            max_grad_norm
                        )
                        else 0.0
                    )


                    # ---------------------------------------------------------
                    # OPTIMIZER STEP
                    # ---------------------------------------------------------

                    if scaler_enabled:

                        scaler.step(
                            optimizer
                        )

                        scaler.update()


                    else:

                        optimizer.step()


                    # ---------------------------------------------------------
                    # PPO DIAGNOSTYKA
                    # ---------------------------------------------------------

                    with torch.no_grad():

                        approx_kl = (
                            (
                                ratio
                                - 1.0
                            )
                            -
                            logratio
                        ).mean().item()


                        clipfrac = (
                            (
                                (
                                    ratio
                                    - 1.0
                                )
                                .abs()
                                >
                                clip_coef
                            )
                            .float()
                            .mean()
                            .item()
                        )


                    losses.append(
                        float(
                            loss.item()
                        )
                    )


                    policy_losses.append(
                        float(
                            policy_loss.item()
                        )
                    )


                    value_losses.append(
                        float(
                            value_loss.item()
                        )
                    )


                    entropies.append(
                        float(
                            entropy.item()
                        )
                    )


                    kls.append(
                        float(
                            approx_kl
                        )
                    )


                    clipfracs.append(
                        float(
                            clipfrac
                        )
                    )


                    # ---------------------------------------------------------
                    # TARGET KL
                    # ---------------------------------------------------------

                    if (
                        target_kl > 0
                        and
                        approx_kl
                        >
                        target_kl
                    ):

                        stop_for_kl = True
                        break


                if stop_for_kl:
                    break


            # =================================================================
            # 3) METRYKI
            # =================================================================

            global_step += (
                train_size
            )


            games_total += (
                update_games
            )


            elapsed = max(
                1e-9,
                time.perf_counter()
                -
                update_started,
            )


            denom = max(
                1,
                update_games,
            )


            metrics = {

                "update":
                    update,

                "global_step":
                    global_step,

                "loss":
                    _mean(
                        losses
                    ),

                "policy_loss":
                    _mean(
                        policy_losses
                    ),

                "value_loss":
                    _mean(
                        value_losses
                    ),

                "entropy":
                    _mean(
                        entropies
                    ),

                "approx_kl":
                    _mean(
                        kls
                    ),

                "clip_fraction":
                    _mean(
                        clipfracs
                    ),


                # =============================================================
                # GRADIENTY
                # =============================================================

                "grad_norm_mean":
                    _mean(
                        grad_norms
                    ),

                "grad_norm_p95":
                    _percentile(
                        grad_norms,
                        0.95,
                    ),

                "grad_norm_max":
                    (
                        max(
                            grad_norms
                        )
                        if grad_norms
                        else 0.0
                    ),

                "grad_clip_fraction":
                    _mean(
                        grad_was_clipped
                    ),

                "grad_scale_mean":
                    _mean(
                        grad_clip_scales
                    ),

                "grad_limit":
                    max_grad_norm,


                # =============================================================
                # TRAINING
                # =============================================================

                "learning_rate":
                    lr,

                "temperature":
                    temperature,


                # =============================================================
                # GRY
                # =============================================================

                "games_completed":
                    games_total,

                "games_this_update":
                    update_games,

                "black_win_rate":
                    (
                        update_black
                        / denom
                    ),

                "white_win_rate":
                    (
                        update_white
                        / denom
                    ),

                "draw_rate":
                    (
                        update_draws
                        / denom
                    ),

                "mean_game_length":
                    _mean(
                        game_lengths
                    ),


                # =============================================================
                # COLLECTOR
                # =============================================================

                "completed_positions_this_update":
                    train_size,

                "generated_positions_this_update":
                    generated_positions,

                "discarded_positions_this_update":
                    discarded_positions,

                "unfinished_games_discarded":
                    unfinished_games_discarded,

                "discard_fraction":
                    (
                        discarded_positions
                        /
                        max(
                            1,
                            generated_positions,
                        )
                    ),


                # =============================================================
                # SPEED
                # =============================================================

                "selfplay_positions_per_second":
                    (
                        generated_positions
                        /
                        collection_elapsed
                    ),

                "positions_per_second":
                    (
                        train_size
                        /
                        elapsed
                    ),


                # =============================================================
                # GPU
                # =============================================================

                "gpu_memory_gb":
                    (
                        torch.cuda
                        .max_memory_allocated(
                            device
                        )
                        / 1e9

                        if (
                            device.type
                            == "cuda"
                        )

                        else 0.0
                    ),
            }


            logger.log(
                metrics,

                write_dashboard=(
                    update
                    %
                    dashboard_every
                    ==
                    0
                ),
            )


            # =================================================================
            # KONSOLA
            # =================================================================

            print(

                f"u={update:06d} "

                f"step="
                f"{global_step:,} "

                f"loss="
                f"{metrics['loss']:.4f} "

                f"ent="
                f"{metrics['entropy']:.3f} "

                f"games="
                f"{update_games:4d} "

                f"complete="
                f"{train_size:,} "

                f"discard="
                f"{discarded_positions:,} "

                f"B/W/D="
                f"{metrics['black_win_rate']:.2f}/"
                f"{metrics['white_win_rate']:.2f}/"
                f"{metrics['draw_rate']:.2f} "

                f"grad="
                f"{metrics['grad_norm_mean']:.2f}/"

                f"p95="
                f"{metrics['grad_norm_p95']:.2f} "

                f"gclip="
                f"{metrics['grad_clip_fraction']:.0%} "

                f"selfplay="
                f"{metrics['selfplay_positions_per_second']:,.0f} "
                f"pos/s "

                f"total="
                f"{metrics['positions_per_second']:,.0f} "
                f"train-pos/s"
            )


            # =================================================================
            # CHECKPOINT
            # =================================================================

            if (
                (update + 1)
                %
                checkpoint_every
                ==
                0
            ):

                path = (
                    checkpoint_mgr.save(
                        update=update,

                        model=model,

                        optimizer=optimizer,

                        config=cfg,

                        global_step=
                            global_step,

                        scaler_state=(
                            scaler.state_dict()
                            if scaler_enabled
                            else None
                        ),

                        extra={
                            "metrics":
                                metrics
                        },
                    )
                )


                print(
                    f"[checkpoint] "
                    f"{path.name}"
                )


        # =====================================================================
        # KOŃCOWY CHECKPOINT
        # =====================================================================

        final_update = max(
            start_update,
            total_updates - 1,
        )


        checkpoint_mgr.save(
            update=final_update,

            model=model,

            optimizer=optimizer,

            config=cfg,

            global_step=
                global_step,

            scaler_state=(
                scaler.state_dict()
                if scaler_enabled
                else None
            ),

            extra={
                "final":
                    True
            },
        )


        logger.write_dashboard()


    # =========================================================================
    # CTRL+C
    # =========================================================================

    except KeyboardInterrupt:

        print(
            "\n[przerwano] "
            "Zapisywanie najnowszego "
            "checkpointu przed wyjściem..."
        )


        checkpoint_mgr.save(

            update=max(
                start_update,

                update
                if "update" in locals()
                else 0,
            ),

            model=model,

            optimizer=optimizer,

            config=cfg,

            global_step=
                global_step,

            scaler_state=(
                scaler.state_dict()
                if scaler_enabled
                else None
            ),

            extra={
                "interrupted":
                    True
            },
        )


        logger.write_dashboard()


        print(
            "[przerwano] Zapisano."
        )



# =============================================================================
# SYMETRIE - POMOCNICZE
# =============================================================================

def _transform_board_actions(
    boards: torch.Tensor,
    actions: torch.Tensor,
    k: int,
    flip: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    """Transformacja D4 do testów.

    Nie jest obecnie używana podczas PPO,
    ponieważ old_logprob odpowiada
    oryginalnemu stanowi i akcji.
    """

    n = boards.shape[-1]


    out = (
        torch.rot90(
            boards,
            k=k,
            dims=(-2, -1),
        )

        if k

        else boards
    )


    r = torch.div(
        actions,
        n,
        rounding_mode="floor",
    )


    c = actions.remainder(
        n
    )


    if k == 1:

        r, c = (
            n - 1 - c,
            r,
        )


    elif k == 2:

        r, c = (
            n - 1 - r,
            n - 1 - c,
        )


    elif k == 3:

        r, c = (
            c,
            n - 1 - r,
        )


    if flip:

        out = torch.flip(
            out,
            dims=(-1,),
        )

        c = (
            n
            - 1
            - c
        )


    return (
        out,
        r * n + c,
    )



# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Trening Connect6 self-play PPO "
            "na terminalnych segmentach "
            "i wektorowych planszach GPU"
        )
    )


    parser.add_argument(
        "--config",
        default="configs/train.yaml",
    )


    args = parser.parse_args()


    train(
        args.config
    )



if __name__ == "__main__":
    main() 