"""Pure preprocessing tests for the frozen PPO Rep-B candidate and source Rep-B."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "submissions" / "ppo_rep_b_500k" / "policy.py"


class _FakePPO:
    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    def predict(self, features, deterministic=True):
        raise AssertionError("predict must not be called by preprocessing test")


def _load_candidate():
    import types

    fake = types.ModuleType("stable_baselines3")
    fake.PPO = _FakePPO
    old = sys.modules.get("stable_baselines3")
    sys.modules["stable_baselines3"] = fake
    try:
        spec = importlib.util.spec_from_file_location(
            "ppo_rep_b_candidate_test",
            CANDIDATE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("stable_baselines3", None)
        else:
            sys.modules["stable_baselines3"] = old


def test_candidate_representation_b_shape_and_dtype():
    module = _load_candidate()
    observation = {
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

    features = module._flatten_observation_representation_b(
        observation
    )

    assert features.shape == (76,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_candidate_uses_exact_lead_times():
    module = _load_candidate()
    assert module._REFERENCE_LEAD_TIMES.tolist() == [3, 2, 1]
    assert module._INVENTORY_SCALE == 200.0
    assert module._DAY_SCALE == 49.0
