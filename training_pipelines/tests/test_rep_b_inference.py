"""Tests for the canonical Representation B preprocessing implementation."""
from __future__ import annotations

import numpy as np

from training_pipelines.src.features.engineered import (
    REPRESENTATION_B_DIM,
    _REFERENCE_LEAD_TIMES,
    flatten_observation_representation_b,
)


def _observation():
    return {
        "inventory": np.asarray([100, 120, 80], dtype=np.int32),
        "arrival_pipeline": np.zeros((3, 4), dtype=np.int32),
        "demand_history": np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [30, 25, 35],
                [31, 24, 36],
                [29, 26, 34],
                [30, 25, 35],
                [32, 25, 36],
            ],
            dtype=np.int32,
        ),
        "day": np.asarray([7], dtype=np.int32),
        "capacity_utilisation": np.asarray([0.5], dtype=np.float32),
    }


def test_representation_b_shape_and_dtype():
    features = flatten_observation_representation_b(_observation())

    assert features.shape == (REPRESENTATION_B_DIM,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_representation_b_uses_exact_lead_times():
    assert _REFERENCE_LEAD_TIMES.tolist() == [3, 2, 1]


def test_representation_b_ignores_early_zero_padding():
    observation = _observation()
    observation["day"] = np.asarray([3], dtype=np.int32)
    features = flatten_observation_representation_b(observation)

    # The 3-day mean features start after the 38 raw features, with 10
    # engineered values per product; check the first product's 7-day mean.
    first_product_seven_day_mean = features[38 + 2]
    assert np.isclose(first_product_seven_day_mean, 30.0 / 200.0, atol=1e-6)
