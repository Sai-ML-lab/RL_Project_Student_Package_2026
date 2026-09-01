
"""Checkpoint save/load helpers shared by the Phase 4+ custom PyTorch algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict,
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state": model_state,
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "hidden_sizes": list(hidden_sizes),
    }
    payload.update(extra or {})
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)