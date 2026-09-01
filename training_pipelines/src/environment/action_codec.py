
"""Single source of truth for action conversion (Phase 1, section 4.1).

The official environment action is ``MultiDiscrete([11, 11, 11])``: index
``a_i`` means an order quantity of ``10 * a_i`` units for product ``i``. DQN
family algorithms instead act over one joint ``Discrete(1331)`` action space
using the base-11 encoding below. Every algorithm and policy submission must
go through these functions -- never re-derive the encoding locally.

Encoding:      joint_index = a1 * 121 + a2 * 11 + a3
Inverse:       a1 = joint_index // 121
               remainder = joint_index % 121
               a2 = remainder // 11
               a3 = remainder % 11
"""
from __future__ import annotations

from typing import Iterable

NUM_PRODUCTS = 3
INDICES_PER_PRODUCT = 11
QUANTITY_STEP = 10
MAX_QUANTITY = 100
JOINT_ACTION_SIZE = INDICES_PER_PRODUCT**NUM_PRODUCTS  # 1331


def _as_int_list(values: Iterable, expected_len: int, label: str) -> list[int]:
    values = list(values)
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} {label}, got {len(values)}.")
    return [int(v) for v in values]


def indices_to_quantities(action_indices: Iterable[int]) -> list[int]:
    """Convert [0..10, 0..10, 0..10] indices to quantities [0, 10, ..., 100]."""
    indices = _as_int_list(action_indices, NUM_PRODUCTS, "action indices")
    for index in indices:
        if not 0 <= index <= INDICES_PER_PRODUCT - 1:
            raise ValueError(f"Action index {index} out of range [0, 10].")
    return [index * QUANTITY_STEP for index in indices]


def quantities_to_indices(order_quantities: Iterable[int]) -> list[int]:
    """Convert official order quantities [0, 10, ..., 100] to action indices."""
    quantities = _as_int_list(order_quantities, NUM_PRODUCTS, "order quantities")
    indices = []
    for quantity in quantities:
        if quantity < 0 or quantity > MAX_QUANTITY or quantity % QUANTITY_STEP != 0:
            raise ValueError(f"Order quantity {quantity} is not one of 0, 10, ..., 100.")
        indices.append(quantity // QUANTITY_STEP)
    return indices


def joint_index_to_multidiscrete(joint_index: int) -> list[int]:
    """Convert one of 1331 joint actions to three indices [a1, a2, a3]."""
    joint_index = int(joint_index)
    if not 0 <= joint_index < JOINT_ACTION_SIZE:
        raise ValueError(f"joint_index must be in [0, {JOINT_ACTION_SIZE}).")
    a1 = joint_index // 121
    remainder = joint_index % 121
    a2 = remainder // 11
    a3 = remainder % 11
    return [a1, a2, a3]


def multidiscrete_to_joint_index(action_indices: Iterable[int]) -> int:
    """Convert three indices [a1, a2, a3] to one joint action index."""
    a1, a2, a3 = _as_int_list(action_indices, NUM_PRODUCTS, "action indices")
    for index in (a1, a2, a3):
        if not 0 <= index <= INDICES_PER_PRODUCT - 1:
            raise ValueError(f"Action index {index} out of range [0, 10].")
    return a1 * 121 + a2 * 11 + a3


def joint_index_to_quantities(joint_index: int) -> list[int]:
    """Compose joint_index_to_multidiscrete + indices_to_quantities."""
    return indices_to_quantities(joint_index_to_multidiscrete(joint_index))


def quantities_to_joint_index(order_quantities: Iterable[int]) -> int:
    """Compose quantities_to_indices + multidiscrete_to_joint_index."""
    return multidiscrete_to_joint_index(quantities_to_indices(order_quantities))