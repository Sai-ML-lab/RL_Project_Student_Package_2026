

"""PPO submission for the IITM RL Inventory Control project.

Technique: Proximal Policy Optimization (PPO)

Loads a Stable-Baselines3 PPO model at import time and performs deterministic
inference in `run_policy`. The observation-flattening code is inlined so that
the submission does not depend on any local training package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Constants mirrored from the environment.
# ---------------------------------------------------------------------------
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)


def _flatten_observation(observation) -> np.ndarray:
    """Reproduce the training-time observation-flattening exactly."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    last3_mean = demand_history[-3:].mean(axis=0)
    last7_mean = demand_history.mean(axis=0)
    last7_std = demand_history.std(axis=0)

    inv_position_units = inventory + pipeline.sum(axis=1)
    inv_position_volume = inv_position_units * _PRODUCT_VOLUMES

    return np.concatenate(
        [
            inventory / 100.0,
            pipeline.reshape(-1) / 100.0,
            last3_mean / _REF_DEMAND_MEANS,
            last7_mean / _REF_DEMAND_MEANS,
            last7_std / _REF_DEMAND_MEANS,
            np.asarray([day / 50.0], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
            inv_position_units / 200.0,
            inv_position_volume / _CAPACITY,
            (inv_position_units - last3_mean) / 100.0,
        ],
        dtype=np.float32,
    ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Load the model once at import time.
# ---------------------------------------------------------------------------
_MODEL_PATH = Path(__file__).resolve().parent / "model.zip"
_MODEL = PPO.load(str(_MODEL_PATH), device="cpu")


def run_policy(observation):
    """Deterministic PPO inference producing three order quantities."""
    features = _flatten_observation(observation)
    action, _ = _MODEL.predict(features, deterministic=True)
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    quantities = (action[:3] * 10).tolist()
    return [int(q) for q in quantities]