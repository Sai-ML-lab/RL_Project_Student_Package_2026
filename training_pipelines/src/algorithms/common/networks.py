
"""Joint-action Q-network (Phase 4, section 7.1)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (256, 256, 128)


class QNetwork(nn.Module):
    """MLP mapping a flat observation to Q-values over the joint action space."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))