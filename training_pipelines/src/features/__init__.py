
"""Observation feature representations shared by every algorithm (Phase 1, section 4.2)."""
from .engineered import (
    ENGINEERED_FEATURE_DIM,
    REPRESENTATION_B_DIM,
    EngineeredObsWrapper,
    flatten_observation_engineered,
    flatten_observation_representation_b,
)
from .normalizer import DEFAULT_SCALES, NormalizationScales
from .observation import RAW_FEATURE_DIM, RawObsWrapper, flatten_observation_raw

__all__ = [
    "DEFAULT_SCALES",
    "ENGINEERED_FEATURE_DIM",
    "RAW_FEATURE_DIM",
    "REPRESENTATION_B_DIM",
    "EngineeredObsWrapper",
    "NormalizationScales",
    "RawObsWrapper",
    "flatten_observation_engineered",
    "flatten_observation_raw",
    "flatten_observation_representation_b",
]