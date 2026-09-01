

"""Neural Network SARSA submission (v3, iteration-3 "RL Final Run" doc,
Phase 5 -- true SARSA target, replay buffer stores the ACTUAL next action).

Technique: Neural Network SARSA (joint 1331-action Q-network), config
sarsa_c (learning_rate=3e-4, gamma=0.98, epsilon_final=0.03), promoted seed
20260825, 500,000 transitions. Uses Representation B (raw + engineered,
76-dim -- FIXED zero-padding + days-of-supply formulas vs the v1 submission).

At inference time the SARSA target collapses to plain greedy argmax over
Q(s, .), decoded from the joint action index via the same base-11 encoding
used by the DQN-family submissions.

The observation-flattening code is inlined so this submission does not
depend on the local `src` package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_HORIZON = 50.0
_REFERENCE_LEAD_TIMES = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
_DEMAND_HISTORY_DAYS = 7

_INVENTORY_SCALE = 200.0
_PIPELINE_SCALE = 100.0
_DEMAND_HISTORY_SCALE = 100.0
_DAY_SCALE = 49.0


def _flatten_raw(observation) -> np.ndarray:
    """Representation A: 38 raw, normalized features."""
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
    """38 engineered features: 30 per-product + 8 global."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    inventory_position = inventory + pipeline.sum(axis=1)

    # `demand_history` is zero-padded at the start of an episode -- only the
    # last `valid_days` rows are real observations; ignore the padding.
    valid_days = min(day, _DEMAND_HISTORY_DAYS)
    if valid_days == 0:
        last3_mean = np.zeros(3, dtype=np.float32)
        last7_mean = np.zeros(3, dtype=np.float32)
        demand_std = np.zeros(3, dtype=np.float32)
    else:
        valid_history = demand_history[-valid_days:]
        valid3 = valid_history[-min(valid_days, 3):]
        last3_mean = valid3.mean(axis=0)
        last7_mean = valid_history.mean(axis=0)
        demand_std = valid_history.std(axis=0)
    demand_trend = last3_mean - last7_mean

    within_lead_time = np.zeros(3, dtype=np.float32)
    after_lead_time = np.zeros(3, dtype=np.float32)
    for product in range(3):
        lead_time = int(_REFERENCE_LEAD_TIMES[product])
        within_lead_time[product] = pipeline[product, :lead_time].sum()
        after_lead_time[product] = pipeline[product, lead_time:].sum()

    lead_time_demand_estimate = last7_mean * _REFERENCE_LEAD_TIMES
    inventory_position_gap = inventory_position - lead_time_demand_estimate
    estimated_days_of_supply = inventory_position / np.maximum(last7_mean, 1.0)

    per_product = np.stack(
        [
            inventory_position, last3_mean, last7_mean, demand_trend, demand_std,
            within_lead_time, after_lead_time, lead_time_demand_estimate,
            inventory_position_gap, estimated_days_of_supply,
        ],
        axis=1,
    ).reshape(-1).astype(np.float32)

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
        [current_volume_ratio, headroom_ratio, pipeline_volume_ratio, projected_utilisation,
         remaining_fraction, early, middle, late],
        dtype=np.float32,
    )
    return np.concatenate([per_product, global_features]).astype(np.float32)


def _flatten_observation(observation) -> np.ndarray:
    """Representation B: raw (38) + engineered (38) = 76 features."""
    return np.concatenate(
        [_flatten_raw(observation), _flatten_engineered(observation)]
    ).astype(np.float32)


def _decode_joint_index(index: int) -> list[int]:
    """Base-11 decode into three order quantities in {0, 10, ..., 100}
    (joint_index = a1*121 + a2*11 + a3, see src/environment/action_codec.py)."""
    index = int(index)
    a3 = index % 11
    index //= 11
    a2 = index % 11
    a1 = index // 11
    return [a1 * 10, a2 * 10, a3 * 10]


class _QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes) -> None:
        super().__init__()
        layers = []
        last = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


_CKPT_PATH = Path(__file__).resolve().parent / "policy_state.pt"
_CKPT = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)

_MODEL = _QNetwork(int(_CKPT["obs_dim"]), int(_CKPT["n_actions"]), list(_CKPT["hidden_sizes"]))
_MODEL.load_state_dict(_CKPT["model_state"])
_MODEL.eval()


def run_policy(observation):
    """Deterministic (greedy) Neural SARSA inference producing three order quantities."""
    features = _flatten_observation(observation)
    with torch.no_grad():
        q_values = _MODEL(torch.from_numpy(features).unsqueeze(0))
    joint_index = int(torch.argmax(q_values, dim=-1).item())
    return _decode_joint_index(joint_index)