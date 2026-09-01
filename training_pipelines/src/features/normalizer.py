
"""Fixed, environment-aware normalization scales (Phase 1, section 4.2).

Scales are constants, never statistics fit on leaderboard episodes, so
training-time and inference-time normalization are always identical.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizationScales:
    """Divisors applied to raw observation fields (Representation A)."""

    inventory_scale: float = 200.0
    pipeline_scale: float = 100.0
    demand_history_scale: float = 100.0
    day_scale: float = 49.0

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationScales":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


DEFAULT_SCALES = NormalizationScales()