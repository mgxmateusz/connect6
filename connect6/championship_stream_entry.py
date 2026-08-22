from __future__ import annotations

import argparse

import torch

from . import championship_stream as stream

_legacy = stream._legacy
base = stream.base


@torch.inference_mode()
def _fixed_step(
    self: stream.GlobalTableScheduler,
    sync_interval: int,
) -> tuple[list[tuple[int, stream.GameJob, int]], list[int]]:
    if self.active_count == 0:
        return self.sync()

    x_all = self.env.network_input()
    legal_all = self.env.legal_mask()
    idx = self.active_indices
    x = x_all.index_select(0, idx)
    legal = legal_all.index_select(0, idx)

    players = self.env.current_player.index_select(0, idx)
    black_ids = self.black_model_ids.index_select(0, idx)
    white_ids = self.white_model_ids.index_select(0, idx)
    actor_ids = torch.where(players.eq(1), black_ids, white_ids)

    with _legacy._autocast_context(self.device, self.amp, self.amp_dtype):
        logits = self.ensemble.forward_indexed_direct(x, actor_ids)

    if self.temperature <= 0:
        # Argmax nie potrzebuje FP32. Stary helper wykonywał logits.float() dla
        # całego [B,361] przy każdym ruchu; tutaj maskujemy bez zmiany dtype.
        chosen = logits.masked_fill(~legal, float("-inf")).argmax(dim=1)
    else:
        chosen = _legacy._choose_actions(
            logits, legal, self.temperature, self.generator
        )

    # Inactive sloty są zamrożone przez active mask, więc nie potrzebujemy już
    # kosztownego legal_all.to(int8).argmax(361) tylko po to, by stworzyć dummy.
    full_actions = torch.zeros(self.tables, dtype=torch.long, device=self.device)
    full_actions.index_copy_(0, idx, chosen)

    fast_step = getattr(self.env, "masked_step", None)
    if fast_step is not None:
        done, winner = fast_step(full_actions, self.active)
    else:
        done, winner = base._masked_step(self.env, full_actions, self.active)

    newly_done = self.active & done
    self.winners.copy_(torch.where(newly_done, winner, self.winners))
    self.active.logical_and_(~done)
    self._moves_since_sync += 1

    if self._moves_since_sync >= max(1, int(sync_interval)):
        return self.sync()
    return [], []


stream.GlobalTableScheduler.step = _fixed_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="King of Connect6 — fast all-resident championship"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()

    from . import championship_fast_run

    championship_fast_run.run(args.config)


if __name__ == "__main__":
    main()
