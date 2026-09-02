from __future__ import annotations

import numpy as np

from industrial_inventory_env import IndustrialInventoryEnv
from training_pipelines.src.features.representation_c import (
    REPRESENTATION_C_ADDITION_DIM,
    REPRESENTATION_C_DIM,
    compute_representation_c_additions,
    flatten_observation_representation_c,
)
from training_pipelines.training_utils.env_factory import load_assigned_config


def _make_env() -> IndustrialInventoryEnv:
    return IndustrialInventoryEnv(
        student_config=load_assigned_config(),
        scenario_mode="stationary",
        domain_randomization=False,
    )


def test_representation_c_shape_and_finiteness() -> None:
    env = _make_env()
    try:
        obs, _ = env.reset(seed=900)
        features = flatten_observation_representation_c(obs)
        additions = compute_representation_c_additions(obs)
        assert features.shape == (REPRESENTATION_C_DIM,)
        assert additions.shape == (REPRESENTATION_C_ADDITION_DIM,)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
    finally:
        env.close()


def test_early_padding_does_not_create_fake_demand() -> None:
    env = _make_env()
    try:
        obs, _ = env.reset(seed=901)
        additions_day0 = compute_representation_c_additions(obs)
        assert np.allclose(additions_day0[:21], 0.0)

        obs, *_ = env.step(np.asarray([0, 0, 0], dtype=np.int64))
        additions_day1 = compute_representation_c_additions(obs)
        assert np.any(np.abs(additions_day1[:3]) > 0.0)
    finally:
        env.close()


def test_representation_c_is_deterministic() -> None:
    env = _make_env()
    try:
        obs, _ = env.reset(seed=902)
        first = flatten_observation_representation_c(obs)
        second = flatten_observation_representation_c(obs)
        assert np.array_equal(first, second)
    finally:
        env.close()
