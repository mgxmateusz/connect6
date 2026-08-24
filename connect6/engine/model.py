from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    """W pełni konwolucyjna sieć POLICY + VALUE dla Connect6.

    Wejście ma kształt [B, 4, H, W]:
      0 = moje kamienie,
      1 = kamienie przeciwnika,
      2 = maska prawdziwej planszy (1 na planszy, 0 w paddingu),
      3 = czy aktualna decyzja jest ostatnim kamieniem w turze.

    Backbone zachowuje rozdzielczość planszy przez cały model. Dla wydajności
    GroupNorm nie jest już wykonywany po każdym convie. Normalizujemy po
    warstwach 0, 2, 5 i 7: po wejściu do każdego nowego poziomu szerokości
    (32/64/96 kanałów) oraz na końcu backbone. Każdy aktywny GroupNorm ma
    dokładnie osiem kanałów na grupę.
    """

    DEFAULT_KERNELS = (23, 3, 3, 3, 3, 3, 3, 3)
    DEFAULT_CHANNELS = (32, 32, 64, 64, 64, 96, 96, 96)
    INPUT_CHANNELS = 4
    CHANNELS_PER_GROUP = 8
    NORM_LAYERS = (0, 2, 5, 7)

    def __init__(
        self,
        board_size: int = 19,
        kernels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size
        self.input_channels = self.INPUT_CHANNELS

        kernels = tuple(self.DEFAULT_KERNELS if kernels is None else kernels)
        channels = tuple(self.DEFAULT_CHANNELS if channels is None else channels)
        if len(kernels) != len(channels):
            raise ValueError("kernels i channels muszą mieć tę samą liczbę elementów")
        if not kernels:
            raise ValueError("CNN musi mieć co najmniej jedną warstwę")
        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("Każdy kernel musi być dodatnią liczbą nieparzystą")
        if any(c <= 0 for c in channels):
            raise ValueError("Liczba kanałów musi być większa od zera")
        if any(channels[i] % self.CHANNELS_PER_GROUP != 0 for i in self.NORM_LAYERS if i < len(channels)):
            raise ValueError(
                f"Warstwy z GroupNorm muszą mieć liczbę kanałów podzielną przez "
                f"{self.CHANNELS_PER_GROUP}"
            )

        self.kernels = kernels
        self.channels = channels
        self.norm_layers = tuple(i for i in self.NORM_LAYERS if i < len(channels))

        convs: list[nn.Conv2d] = []
        norms: list[nn.Module] = []
        in_channels = self.input_channels
        for layer, (kernel, out_channels) in enumerate(zip(kernels, channels)):
            convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    bias=False,
                )
            )
            if layer in self.norm_layers:
                norms.append(
                    nn.GroupNorm(
                        num_groups=out_channels // self.CHANNELS_PER_GROUP,
                        num_channels=out_channels,
                        affine=True,
                    )
                )
            else:
                norms.append(nn.Identity())
            in_channels = out_channels
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.activation = nn.SiLU(inplace=True)

        self.policy_output = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.value_output = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.value_tanh = nn.Tanh()

        self._inicjalizuj_wagi()

    @property
    def receptive_field(self) -> int:
        return 1 + sum(kernel - 1 for kernel in self.kernels)

    def _inicjalizuj_wagi(self) -> None:
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        for norm in self.norms:
            if isinstance(norm, nn.GroupNorm):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

        nn.init.normal_(self.policy_output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy_output.bias)
        nn.init.normal_(self.value_output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_output.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.input_channels, self.board_size, self.board_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Wejście modelu musi mieć kształt [B, {expected[0]}, "
                f"{expected[1]}, {expected[2]}], otrzymano {tuple(x.shape)}."
            )

        features = x
        for conv, norm in zip(self.convs, self.norms):
            features = self.activation(norm(conv(features)))

        logits = self.policy_output(features).flatten(1)
        value_map = self.value_output(features)
        value = self.value_tanh(value_map.mean(dim=(-2, -1))).squeeze(1)
        return logits, value


def build_model(model_cfg: dict[str, Any], board_size: int) -> PolicyValueNet:
    cfg = dict(model_cfg)
    cfg.pop("compile", None)
    cfg.pop("compile_mode", None)
    cfg.pop("architecture_version", None)
    return PolicyValueNet(board_size=board_size, **cfg)


def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
