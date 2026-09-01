
"""DQN submission for the IITM RL Inventory Control project.

Technique: Deep Q-Network (DQN)

Trained with a flattened Discrete(1331) action space (base-11 encoding of the
underlying MultiDiscrete([11,11,11]) space). Decoded here into three
per-product order quantities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)


def _flatten_observation(observation) -> np.ndarray:
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


def _decode_joint_index(index: int) -> list[int]:
    """Base-11 decode into three order quantities in {0, 10, ..., 100}."""
    index = int(index)
    a2 = index % 11
    index //= 11
    a1 = index % 11
    a0 = index // 11
    return [a0 * 10, a1 * 10, a2 * 10]


_MODEL_PATH = Path(__file__).resolve().parent / "model.zip"
_MODEL = DQN.load(str(_MODEL_PATH), device="cpu")


def run_policy(observation):
    features = _flatten_observation(observation)
    action, _ = _MODEL.predict(features, deterministic=True)
    joint_index = int(np.asarray(action).reshape(-1)[0])
    return _decode_joint_index(joint_index)