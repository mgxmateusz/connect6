from __future__ import annotations

from typing import Any

import torch
from torch import nn


# =============================================================================
# MODEL MLP CONNECT6
# =============================================================================
#
# Sieć dostaje JEDEN zwykły wektor liczb.
# Dla standardowej planszy 19x19 wejście ma 724 wartości:
#
#   361 wartości = moje kamienie
#   361 wartości = kamienie przeciwnika
#     1 wartość   = ile kamieni zostało do postawienia w tej turze / 2
#     1 wartość   = czy aktualny gracz gra czarnymi
#
# Razem:
#   361 + 361 + 1 + 1 = 724
#
# Dalej jest zwykłe MLP, np.:
#
#   724 -> 1024 -> 512 -> 256 -> 128
#
# Po warstwach wspólnych sieć rozdziela się na dwie końcówki:
#
#   POLICY -> 361 logitów, po jednym dla każdego pola planszy
#   VALUE  -> 1 wartość oceniającą pozycję w zakresie [-1, +1]
#
# To nadal jest JEDEN model. Dwie końcówki są potrzebne dlatego, że PPO
# jednocześnie uczy:
#   1) który ruch wybrać,
#   2) jak dobra jest aktualna pozycja.
#
# Całą architekturę warstw ukrytych ustawiasz w configs/train.yaml.
# Każda warstwa może mieć własną liczbę neuronów, normalizację, aktywację
# i dropout.
# =============================================================================


def _aktywacja(nazwa: str | None) -> nn.Module:
    """Tworzy aktywację wybraną w pliku konfiguracyjnym."""
    nazwa = "none" if nazwa is None else str(nazwa).lower()

    if nazwa in ("none", "identity", "off"):
        return nn.Identity()
    if nazwa == "silu":
        return nn.SiLU(inplace=True)
    if nazwa == "gelu":
        return nn.GELU()
    if nazwa == "relu":
        return nn.ReLU(inplace=True)
    if nazwa == "tanh":
        return nn.Tanh()
    if nazwa == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(
        f"Nieznana aktywacja: {nazwa}. "
        "Dostępne: silu | gelu | relu | tanh | sigmoid | none"
    )


def _normalizacja(nazwa: str | None, liczba_neuronow: int) -> nn.Module:
    """Tworzy normalizację dla zwykłej warstwy Linear."""
    nazwa = "none" if nazwa is None else str(nazwa).lower()

    if nazwa in ("none", "identity", "off"):
        return nn.Identity()
    if nazwa == "layer":
        return nn.LayerNorm(liczba_neuronow)
    if nazwa == "batch":
        return nn.BatchNorm1d(liczba_neuronow)

    raise ValueError(
        f"Nieznana normalizacja: {nazwa}. Dostępne: layer | batch | none"
    )


class WarstwaMLP(nn.Module):
    """Jedna zwykła, w pełni konfigurowalna warstwa MLP."""

    def __init__(self, wejscia: int, konfiguracja: dict[str, Any]) -> None:
        super().__init__()

        neurony = int(konfiguracja["neurons"])
        norm = str(konfiguracja.get("norm", "none"))
        activation = str(konfiguracja.get("activation", "silu"))
        dropout = float(konfiguracja.get("dropout", 0.0))

        if neurony <= 0:
            raise ValueError("Liczba neuronów musi być większa od zera.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout musi należeć do przedziału [0.0, 1.0).")

        # Gdy bezpośrednio po Linear jest normalizacja, bias jest zwykle zbędny.
        uzyj_bias = norm.lower() in ("none", "identity", "off")

        elementy: list[nn.Module] = [
            nn.Linear(wejscia, neurony, bias=uzyj_bias),
            _normalizacja(norm, neurony),
            _aktywacja(activation),
        ]
        if dropout > 0.0:
            elementy.append(nn.Dropout(dropout))

        self.warstwa = nn.Sequential(*elementy)
        self.liczba_wyjsc = neurony

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.warstwa(x)


def _zbuduj_stos_warstw(
    liczba_wejsc: int,
    konfiguracje: list[dict[str, Any]],
    *,
    nazwa: str,
) -> tuple[nn.Sequential, int]:
    """Buduje listę zwykłych warstw Linear dokładnie w kolejności z YAML."""
    warstwy: list[nn.Module] = []
    aktualny_rozmiar = int(liczba_wejsc)

    for indeks, konfiguracja in enumerate(konfiguracje):
        if "neurons" not in konfiguracja:
            raise ValueError(f"Brak pola 'neurons' w {nazwa}[{indeks}].")

        warstwa = WarstwaMLP(aktualny_rozmiar, konfiguracja)
        warstwy.append(warstwa)
        aktualny_rozmiar = warstwa.liczba_wyjsc

    return nn.Sequential(*warstwy), aktualny_rozmiar


class PolicyValueNet(nn.Module):
    """Jedna zwykła sieć MLP z wyjściem POLICY i VALUE."""

    def __init__(
        self,
        board_size: int = 19,
        layers: list[dict[str, Any]] | None = None,
        policy_layers: list[dict[str, Any]] | None = None,
        value_layers: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()

        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size
        self.input_size = self.action_size * 2 + 2

        # ---------------------------------------------------------------------
        # GŁÓWNA CZĘŚĆ SIECI
        # ---------------------------------------------------------------------
        # Tutaj edytujesz podstawowy ciąg warstw, np.:
        #   724 -> 1024 -> 512 -> 256 -> 128
        # ---------------------------------------------------------------------
        if layers is None:
            layers = [
                {"neurons": 1024, "norm": "layer", "activation": "silu", "dropout": 0.0},
                {"neurons": 512, "norm": "none", "activation": "silu", "dropout": 0.0},
                {"neurons": 256, "norm": "layer", "activation": "silu", "dropout": 0.0},
                {"neurons": 128, "norm": "none", "activation": "silu", "dropout": 0.0},
            ]

        self.layers, rozmiar_wspolny = _zbuduj_stos_warstw(
            self.input_size,
            layers,
            nazwa="layers",
        )

        # ---------------------------------------------------------------------
        # OPCJONALNE WARSTWY TYLKO DLA POLICY
        # ---------------------------------------------------------------------
        # Jeśli lista jest pusta, POLICY wychodzi bezpośrednio z ostatniej
        # warstwy wspólnej. Możesz tu dopisać własne dodatkowe warstwy.
        # ---------------------------------------------------------------------
        if policy_layers is None:
            policy_layers = []
        self.policy_layers, rozmiar_policy = _zbuduj_stos_warstw(
            rozmiar_wspolny,
            policy_layers,
            nazwa="policy_layers",
        )
        self.policy_output = nn.Linear(rozmiar_policy, self.action_size)

        # ---------------------------------------------------------------------
        # OPCJONALNE WARSTWY TYLKO DLA VALUE
        # ---------------------------------------------------------------------
        # VALUE może mieć własne dodatkowe warstwy, ponieważ jego zadaniem jest
        # ocena całej pozycji, a nie wskazanie konkretnego pola.
        # ---------------------------------------------------------------------
        if value_layers is None:
            value_layers = [
                {"neurons": 64, "norm": "none", "activation": "silu", "dropout": 0.0},
            ]
        self.value_layers, rozmiar_value = _zbuduj_stos_warstw(
            rozmiar_wspolny,
            value_layers,
            nazwa="value_layers",
        )
        self.value_output = nn.Linear(rozmiar_value, 1)
        self.value_tanh = nn.Tanh()

        self._inicjalizuj_wagi()

    def _inicjalizuj_wagi(self) -> None:
        """Inicjalizuje wszystkie zwykłe warstwy Linear."""
        for modul in self.modules():
            if isinstance(modul, nn.Linear):
                # Kaiming jest praktycznym wyborem także przy SiLU.
                nn.init.kaiming_normal_(modul.weight, nonlinearity="relu")
                if modul.bias is not None:
                    nn.init.zeros_(modul.bias)

        # Mniejsze wagi na samym wyjściu POLICY ograniczają przypadkowo bardzo
        # ostre preferencje ruchów na samym początku treningu.
        nn.init.normal_(self.policy_output.weight, mean=0.0, std=0.01)
        if self.policy_output.bias is not None:
            nn.init.zeros_(self.policy_output.bias)

        nn.init.normal_(self.value_output.weight, mean=0.0, std=1.0)
        if self.value_output.bias is not None:
            nn.init.zeros_(self.value_output.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Wejście modelu musi mieć kształt [B, {self.input_size}], "
                f"otrzymano {tuple(x.shape)}."
            )

        wspolne = self.layers(x)

        policy_cechy = self.policy_layers(wspolne)
        logits = self.policy_output(policy_cechy)

        value_cechy = self.value_layers(wspolne)
        value = self.value_tanh(self.value_output(value_cechy)).squeeze(-1)

        return logits, value


def build_model(model_cfg: dict[str, Any], board_size: int) -> PolicyValueNet:
    """Buduje model na podstawie sekcji 'model' z pliku YAML."""
    cfg = dict(model_cfg)

    # Te ustawienia dotyczą uruchamiania modelu, a nie jego konstrukcji.
    cfg.pop("compile", None)
    cfg.pop("compile_mode", None)
    cfg.pop("architecture_version", None)

    # Jeśli ktoś przypadkiem użyje starego configu CNN, dostanie jasny błąd.
    stare_pola = {
        "board_layers",
        "global_layers",
        "fusion_layers",
        "channels",
        "residual_blocks",
        "normalization",
    }
    if any(pole in cfg for pole in stare_pola):
        raise ValueError(
            "Wykryto konfigurację starego modelu CNN. Ta wersja projektu używa "
            "wyłącznie zwykłego MLP. Użyj aktualnego configs/train.yaml."
        )

    return PolicyValueNet(board_size=board_size, **cfg)


def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Blokuje zajęte pola, aby softmax nigdy nie mógł ich wylosować."""
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
