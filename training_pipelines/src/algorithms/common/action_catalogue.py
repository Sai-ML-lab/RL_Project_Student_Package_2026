
"""Reduced legal action catalogue for TD(lambda)/Tabular SARSA (Phase 5, section 8.2).

The full 1331-action joint space is intractable for a coarse-coded linear
model with per-action weights, so we curate 40-100 "operationally sensible"
quantity combinations from {0,20,40,60,80,100} per product instead of the
full 6**3=216 cross product.
"""
from __future__ import annotations

from itertools import combinations

QUANTITY_LEVELS = (0, 20, 40, 60, 80, 100)


def build_action_catalogue() -> list[tuple[int, int, int]]:
    """Programmatically curate 40-100 legal joint-quantity combinations."""
    actions: set[tuple[int, int, int]] = set()

    # Uniform orders across all three products.
    for level in QUANTITY_LEVELS:
        actions.add((level, level, level))

    # One product elevated from a zero baseline ("order only what's short").
    for position in range(3):
        for level in QUANTITY_LEVELS[1:]:
            action = [0, 0, 0]
            action[position] = level
            actions.add(tuple(action))

    # One product held back below a high uniform base ("skip the
    # well-stocked product while replenishing the rest").
    high_bases = (40, 60, 80, 100)
    low_values = (0, 20)
    for base in high_bases:
        for position in range(3):
            for low in low_values:
                action = [base, base, base]
                action[position] = low
                actions.add(tuple(action))

    # Two products elevated together, the third at a low baseline ("two
    # products are trending up together").
    elevated_values = (40, 60, 80)
    baselines = (0, 20)
    for pair in combinations(range(3), 2):
        for level in elevated_values:
            for baseline in baselines:
                action = [baseline, baseline, baseline]
                for position in pair:
                    action[position] = level
                actions.add(tuple(action))

    # Explicit staggered examples from the problem statement.
    actions.update(
        {
            (0, 0, 0),
            (20, 20, 20),
            (40, 40, 40),
            (60, 40, 60),
            (40, 20, 60),
            (80, 60, 80),
            (100, 100, 100),
        }
    )

    return sorted(actions)


ACTION_CATALOGUE: list[tuple[int, int, int]] = build_action_catalogue()
CATALOGUE_SIZE = len(ACTION_CATALOGUE)


def catalogue_action_to_env_indices(catalogue_index: int) -> list[int]:
    """Convert a catalogue index to environment action indices in [0, 10]."""
    quantities = ACTION_CATALOGUE[catalogue_index]
    return [quantity // 10 for quantity in quantities]