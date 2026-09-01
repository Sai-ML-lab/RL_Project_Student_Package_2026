
"""TD(lambda) submission for the IITM RL Inventory Control project.

Technique: True Online SARSA(lambda), linear function approximation over
sparse coarse-coded ("banded") features, with a reduced 48-action catalogue.

Self-contained: inlines the band discretizer and the action catalogue so
this submission does not depend on the local `src` package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REFERENCE_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_REFERENCE_LEAD_TIMES = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
_PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
_CAPACITY = 1000.0
_HORIZON = 50.0
_EPS = 1e-6

N_BINS = 3
N_BAND_GROUPS = 18
FEATURE_DIM = N_BAND_GROUPS * N_BINS  # 54

# 48-action reduced catalogue (doc section 8.2) -- must match
# src/algorithms/common/action_catalogue.py's ACTION_CATALOGUE exactly.
ACTION_CATALOGUE = [
    (0, 0, 0), (0, 0, 20), (0, 0, 40), (0, 0, 60), (0, 0, 80), (0, 0, 100),
    (0, 20, 0), (0, 40, 0), (0, 40, 40), (0, 60, 0), (0, 60, 60), (0, 80, 0),
    (0, 80, 80), (0, 100, 0), (0, 100, 100), (20, 0, 0), (20, 20, 20),
    (20, 40, 40), (20, 60, 60), (20, 80, 80), (20, 100, 100), (40, 0, 0),
    (40, 0, 40), (40, 20, 40), (40, 20, 60), (40, 40, 0), (40, 40, 20),
    (40, 40, 40), (60, 0, 0), (60, 0, 60), (60, 20, 60), (60, 40, 60),
    (60, 60, 0), (60, 60, 20), (60, 60, 60), (80, 0, 0), (80, 0, 80),
    (80, 20, 80), (80, 60, 80), (80, 80, 0), (80, 80, 20), (80, 80, 80),
    (100, 0, 0), (100, 0, 100), (100, 20, 100), (100, 100, 0),
    (100, 100, 20), (100, 100, 100),
]


def _coverage_band(days_covered: float, lead_time: float) -> int:
    if days_covered < lead_time:
        return 0
    if days_covered < 2.0 * lead_time:
        return 1
    return 2


def _demand_level_band(recent_mean: float, reference_mean: float) -> int:
    if recent_mean < 0.9 * reference_mean:
        return 0
    if recent_mean > 1.1 * reference_mean:
        return 2
    return 1


def _trend_band(trend: float) -> int:
    if trend < -2.0:
        return 0
    if trend > 2.0:
        return 2
    return 1


def _utilisation_band(value: float) -> int:
    if value < 0.5:
        return 0
    if value < 0.85:
        return 1
    return 2


def _discretize_observation(observation) -> list[int]:
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])

    pipeline_total = pipeline.sum(axis=1)
    inventory_position = inventory + pipeline_total
    last3_mean = demand_history[-3:].mean(axis=0)
    last7_mean = demand_history.mean(axis=0)
    trend = last3_mean - last7_mean
    demand_estimate = np.maximum(last3_mean, _EPS)

    bands: list[int] = []
    for product in range(3):
        lead_time = float(_REFERENCE_LEAD_TIMES[product])
        bands.append(_coverage_band(float(inventory_position[product] / demand_estimate[product]), lead_time))
        bands.append(_coverage_band(float(inventory[product] / demand_estimate[product]), lead_time))
        bands.append(_coverage_band(float(pipeline_total[product] / demand_estimate[product]), lead_time))
        bands.append(_demand_level_band(float(last3_mean[product]), float(_REFERENCE_DEMAND_MEANS[product])))
        bands.append(_trend_band(float(trend[product])))

    current_volume = float(np.dot(inventory, _PRODUCT_VOLUMES))
    pipeline_volume = float(np.dot(pipeline_total, _PRODUCT_VOLUMES))
    projected_utilisation = (current_volume + pipeline_volume) / _CAPACITY
    phase = day / _HORIZON
    episode_phase = 0 if phase < (1.0 / 3.0) else (1 if phase < (2.0 / 3.0) else 2)

    bands.append(_utilisation_band(capacity_utilisation))
    bands.append(_utilisation_band(projected_utilisation))
    bands.append(episode_phase)
    return bands


def _flatten_observation(observation) -> np.ndarray:
    bands = _discretize_observation(observation)
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    for group_index, band in enumerate(bands):
        features[group_index * N_BINS + band] = 1.0
    return features


_WEIGHTS_PATH = Path(__file__).resolve().parent / "policy_weights.npz"
_W = np.load(str(_WEIGHTS_PATH))["W"]


def run_policy(observation):
    """Deterministic (greedy) TD(lambda) inference producing three order quantities."""
    features = _flatten_observation(observation)
    q_values = _W @ features
    action_index = int(np.argmax(q_values))
    return list(ACTION_CATALOGUE[action_index])