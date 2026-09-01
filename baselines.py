
"""Two heuristic baseline policies used as sanity-check floors.

Every RL submission must strictly outperform both baselines on the holdout
evaluation grid; if not, treat it as a bug and investigate.
"""
from __future__ import annotations

import numpy as np

_REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
_LEAD_TIMES = np.asarray([3, 2, 1], dtype=np.float32)


def zero_order_policy(observation: dict) -> list[int]:
    """Never order — the worst-case cost floor."""
    return [0, 0, 0]


def order_up_to_policy(observation: dict) -> list[int]:
    """Generalized order-up-to heuristic using the observation's demand history.

    Target inventory position = max(recent demand mean, reference mean) *
    (lead_time + safety_days). Order the shortfall, rounded up to a multiple
    of 10 and clipped to [0, 100].
    """
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)

    recent = demand_history[-3:].mean(axis=0)
    demand_estimate = np.maximum(recent, _REF_DEMAND_MEANS)

    safety_days = 2.0
    target_position = demand_estimate * (_LEAD_TIMES + safety_days)

    inventory_position = inventory + pipeline.sum(axis=1)
    shortfall = np.maximum(target_position - inventory_position, 0.0)
    order = np.ceil(shortfall / 10.0) * 10.0
    order = np.clip(order, 0.0, 100.0)
    return order.astype(int).tolist()


__all__ = ["order_up_to_policy", "zero_order_policy"]