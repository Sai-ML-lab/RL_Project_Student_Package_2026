"""Double DQN submission for IITM RL Inventory Control.

Technique:
    Double Deep Q-Network (Double DQN) with a dueling joint-action network.

Representation:
    76-dimensional Representation B = 38 normalized raw features +
    38 engineered features.

The policy is self-contained and performs deterministic inference only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_HORIZON = 50.0
_REFERENCE_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.int64)
_NUM_PRODUCTS = 3
_DEMAND_HISTORY_DAYS = 7

_INVENTORY_SCALE = 200.0
_PIPELINE_SCALE = 100.0
_DEMAND_HISTORY_SCALE = 100.0
_DAY_SCALE = 49.0


def _flatten_raw(observation) -> np.ndarray:
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    return np.concatenate(
        [
            inventory / _INVENTORY_SCALE,
            pipeline.reshape(-1) / _PIPELINE_SCALE,
            demand_history.reshape(-1) / _DEMAND_HISTORY_SCALE,
            np.asarray([day / _DAY_SCALE], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def _flatten_engineered(observation) -> np.ndarray:
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    inventory_position = inventory + pipeline.sum(axis=1)

    valid_days = min(day, _DEMAND_HISTORY_DAYS)
    if valid_days == 0:
        last3_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        last7_mean = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
        demand_std = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    else:
        valid_history = demand_history[-valid_days:]
        valid3 = valid_history[-min(valid_days, 3):]
        last3_mean = valid3.mean(axis=0)
        last7_mean = valid_history.mean(axis=0)
        demand_std = valid_history.std(axis=0)

    demand_trend = last3_mean - last7_mean

    within_lead_time = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    after_lead_time = np.zeros(_NUM_PRODUCTS, dtype=np.float32)
    for product in range(_NUM_PRODUCTS):
        lead_time = int(_REFERENCE_LEAD_TIMES[product])
        within_lead_time[product] = pipeline[product, :lead_time].sum()
        after_lead_time[product] = pipeline[product, lead_time:].sum()

    lead_time_demand_estimate = last7_mean * _REFERENCE_LEAD_TIMES
    inventory_position_gap = inventory_position - lead_time_demand_estimate
    estimated_days_of_supply = inventory_position / np.maximum(last7_mean, 1.0)

    per_product = np.stack(
        [
            inventory_position,
            last3_mean,
            last7_mean,
            demand_trend,
            demand_std,
            within_lead_time,
            after_lead_time,
            lead_time_demand_estimate,
            inventory_position_gap,
            estimated_days_of_supply,
        ],
        axis=1,
    ).reshape(-1).astype(np.float32, copy=False)

    current_volume = float(np.dot(inventory, _PRODUCT_VOLUMES))
    pipeline_volume = float(np.dot(pipeline.sum(axis=1), _PRODUCT_VOLUMES))
    current_volume_ratio = current_volume / _CAPACITY
    headroom_ratio = max(_CAPACITY - current_volume, 0.0) / _CAPACITY
    pipeline_volume_ratio = pipeline_volume / _CAPACITY
    projected_utilisation = (current_volume + pipeline_volume) / _CAPACITY
    remaining_fraction = max(_HORIZON - day, 0.0) / _HORIZON

    phase = day / _HORIZON
    early = 1.0 if phase < (1.0 / 3.0) else 0.0
    middle = 1.0 if (1.0 / 3.0) <= phase < (2.0 / 3.0) else 0.0
    late = 1.0 if phase >= (2.0 / 3.0) else 0.0

    global_features = np.asarray(
        [
            current_volume_ratio,
            headroom_ratio,
            pipeline_volume_ratio,
            projected_utilisation,
            remaining_fraction,
            early,
            middle,
            late,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [per_product, global_features]
    ).astype(np.float32, copy=False)


def _features(observation) -> np.ndarray:
    features = np.concatenate(
        [
            _flatten_raw(observation),
            _flatten_engineered(observation),
        ]
    ).astype(np.float32, copy=False)

    if features.shape != (76,):
        raise ValueError(
            f"Expected 76-D Representation B, got {features.shape}"
        )
    return features


class _DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(h2, 1)
        self.advantage_head = nn.Linear(h2, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(obs)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


_CKPT_PATH = Path(__file__).resolve().parent / "policy_state.pt"
_CKPT = torch.load(
    str(_CKPT_PATH),
    map_location="cpu",
    weights_only=False,
)

_MODEL = _DuelingQNetwork(
    int(_CKPT["obs_dim"]),
    int(_CKPT["n_actions"]),
    tuple(int(x) for x in _CKPT["hidden_sizes"]),
)
_MODEL.load_state_dict(_CKPT["model_state"])
_MODEL.eval()


def _decode_joint_index(index: int) -> list[int]:
    index = int(index)
    if not 0 <= index < 1331:
        raise ValueError(f"Invalid joint action index: {index}")

    a3 = index % 11
    index //= 11
    a2 = index % 11
    a1 = index // 11

    quantities = [a1 * 10, a2 * 10, a3 * 10]
    if not all(q in range(0, 101, 10) for q in quantities):
        raise ValueError(f"Invalid decoded quantities: {quantities}")
    return quantities


def run_policy(observation):
    """Return deterministic order quantities for Products 1, 2 and 3."""
    features = _features(observation)

    with torch.no_grad():
        q_values = _MODEL(
            torch.from_numpy(features).unsqueeze(0)
        )
        joint_index = int(q_values.argmax(dim=1).item())

    return _decode_joint_index(joint_index)
