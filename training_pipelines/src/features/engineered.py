
"""Representation B: raw + engineered features (Phase 1, section 4.2; formulas
per the iteration-3 "RL Final Run" doc, Phase 3).

Built only from the documented observation dict -- never from `info` or
future demand. Physical constants (capacity, lead times, volumes) are read
directly from `IndustrialInventoryEnv` so they can never drift out of sync
with the official environment.

Per-product engineered features (10 x 3 products = 30 values, product-major):
    inventory_position, recent_3day_mean_demand, recent_7day_mean_demand,
    demand_trend, demand_std, pipeline_within_lead_time,
    pipeline_after_lead_time, lead_time_demand_estimate,
    inventory_position_gap, estimated_days_of_supply

Global engineered features (8 values):
    current_inventory_volume_ratio, capacity_headroom_ratio,
    pipeline_volume_ratio, projected_capacity_utilisation,
    remaining_episode_fraction, episode_phase_early, episode_phase_middle,
    episode_phase_late

Early-episode demand means (`recent_3day_mean_demand`, `recent_7day_mean_demand`,
`demand_std`) ignore the zero-padded rows of `demand_history` that exist before
enough real days have elapsed -- do not let that padding bias the mean toward
zero.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from industrial_inventory_env.environment import IndustrialInventoryEnv

from .observation import RAW_FEATURE_DIM, flatten_observation_raw

_PRODUCT_VOLUMES = IndustrialInventoryEnv.PRODUCT_VOLUMES.astype(np.float32)
_CAPACITY = float(IndustrialInventoryEnv.CAPACITY)
_HORIZON = float(IndustrialInventoryEnv.HORIZON)
_REFERENCE_LEAD_TIMES = IndustrialInventoryEnv.REFERENCE_LEAD_TIMES.astype(np.float32)
_NUM_PRODUCTS = IndustrialInventoryEnv.NUM_PRODUCTS
_DEMAND_HISTORY_DAYS = IndustrialInventoryEnv.DEMAND_HISTORY_DAYS
_EPS = 1e-6

PER_PRODUCT_FEATURE_NAMES = (
    "inventory_position",
    "recent_3day_mean_demand",
    "recent_7day_mean_demand",
    "demand_trend",
    "demand_std",
    "pipeline_within_lead_time",
    "pipeline_after_lead_time",
    "lead_time_demand_estimate",
    "inventory_position_gap",
    "estimated_days_of_supply",
)
PER_PRODUCT_FEATURE_DIM = len(PER_PRODUCT_FEATURE_NAMES) * _NUM_PRODUCTS  # 30

GLOBAL_FEATURE_NAMES = (
    "current_inventory_volume_ratio",
    "capacity_headroom_ratio",
    "pipeline_volume_ratio",
    "projected_capacity_utilisation",
    "remaining_episode_fraction",
    "episode_phase_early",
    "episode_phase_middle",
    "episode_phase_late",
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)  # 8

ENGINEERED_FEATURE_DIM = PER_PRODUCT_FEATURE_DIM + GLOBAL_FEATURE_DIM  # 38
REPRESENTATION_B_DIM = RAW_FEATURE_DIM + ENGINEERED_FEATURE_DIM  # 76


def compute_per_product_features(observation) -> np.ndarray:
    """Return the 10 per-product engineered features, flattened product-major (30,)."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    inventory_position = inventory + pipeline.sum(axis=1)

    # `demand_history` is a rolling window (oldest first, most recent last) that
    # is zero-padded at the start of an episode; only the last `valid_days` rows
    # are real observations.
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
    )  # shape (num_products, 10)
    return per_product.reshape(-1).astype(np.float32)


def compute_global_features(observation) -> np.ndarray:
    """Return the 8 global engineered features (8,)."""
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])

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

    return np.asarray(
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


def flatten_observation_engineered(observation) -> np.ndarray:
    """Return the 38 engineered features: 30 per-product + 8 global."""
    return np.concatenate(
        [compute_per_product_features(observation), compute_global_features(observation)]
    ).astype(np.float32)


def flatten_observation_representation_b(observation) -> np.ndarray:
    """Representation B: raw (38) + engineered (38) = 76 features."""
    return np.concatenate(
        [flatten_observation_raw(observation), flatten_observation_engineered(observation)]
    ).astype(np.float32)


class EngineeredObsWrapper(gym.ObservationWrapper):
    """Observation wrapper producing the fixed-size Representation B vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(REPRESENTATION_B_DIM,), dtype=np.float32
        )

    def observation(self, observation):
        return flatten_observation_representation_b(observation)