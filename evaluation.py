
"""Local evaluation harness for the IITM RL inventory-control project.

Runs a policy across a grid of (seed x scenario_mode) episodes on the OFFICIAL
(unshaped) environment and returns a tidy DataFrame plus a per-scenario summary.

A "policy" here is any callable `(observation_dict) -> [q1, q2, q3]` returning
quantities in {0, 10, ..., 100}. This is exactly the leaderboard interface, so
the harness can be used against both raw callables and imported `run_policy`
functions from a submission file.

Usage
-----
    from evaluation import evaluate_policy, HOLDOUT_SEEDS, SCENARIO_MODES
    df, summary = evaluate_policy(my_policy, seeds=HOLDOUT_SEEDS, scenario_modes=SCENARIO_MODES)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from industrial_inventory_env import IndustrialInventoryEnv

PolicyFn = Callable[[dict], Iterable[int]]

# Reserved seeds — never used during training.
HOLDOUT_SEEDS: list[int] = list(range(900, 940))
# Smaller set for quick in-training callbacks.
FAST_EVAL_SEEDS: list[int] = list(range(900, 910))

SCENARIO_MODES: list[str] = [
    "stationary",
    "seasonal",
    "trend",
    "shock",
    "random",
]


def _load_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if config is None:
        config = Path(__file__).resolve().parent / "assigned_config.json"
    with open(config, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_action(action: Any) -> list[int]:
    if isinstance(action, np.ndarray):
        action = action.tolist()
    if not isinstance(action, (list, tuple)):
        raise TypeError("Policy must return a list, tuple or numpy array.")
    if len(action) != 3:
        raise ValueError("Policy must return exactly three quantities.")
    out: list[int] = []
    for value in action:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Order quantities must be integers, not bool.")
        integer = int(value)
        if integer not in range(0, 101, 10):
            raise ValueError(f"Order quantity {integer} is not in 0..100 step 10.")
        out.append(integer)
    return out


def _run_episode(
    env: IndustrialInventoryEnv,
    policy: PolicyFn,
    seed: int,
) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=int(seed))
    total_cost = 0.0
    holding = stockout = ordering = discarding = 0.0
    total_demand = 0
    total_fulfilled = 0
    total_discarded = 0
    sum_capacity_util = 0.0
    step_count = 0
    call_time_sum = 0.0

    done = False
    while not done:
        started = time.perf_counter()
        quantities = _validate_action(policy(obs))
        call_time_sum += time.perf_counter() - started

        action_indices = env.quantities_to_action_indices(quantities)
        obs, reward, terminated, truncated, info = env.step(action_indices)

        total_cost += info["costs"]["daily_total"]
        holding += info["costs"]["holding"]
        stockout += info["costs"]["stockout"]
        ordering += info["costs"]["ordering"]
        discarding += info["costs"]["discarding"]

        total_demand += int(np.sum(info["demand"]))
        total_fulfilled += int(np.sum(info["fulfilled_demand"]))
        total_discarded += int(np.sum(info["discarded_units"]))
        sum_capacity_util += float(obs["capacity_utilisation"][0])
        step_count += 1
        done = bool(terminated or truncated)

    service_level = total_fulfilled / max(total_demand, 1)
    return {
        "seed": int(seed),
        "episode_cost": float(total_cost),
        "holding_cost": float(holding),
        "stockout_cost": float(stockout),
        "ordering_cost": float(ordering),
        "discarding_cost": float(discarding),
        "service_level": float(service_level),
        "total_demand": int(total_demand),
        "total_fulfilled": int(total_fulfilled),
        "total_discarded": int(total_discarded),
        "avg_capacity_utilisation": float(sum_capacity_util / max(step_count, 1)),
        "steps": int(step_count),
        "avg_policy_call_ms": float(1000.0 * call_time_sum / max(step_count, 1)),
        "variant_id": reset_info["variant_id"],
        "config_fingerprint": reset_info.get("config_fingerprint"),
    }


def evaluate_policy(
    policy: PolicyFn,
    *,
    seeds: Iterable[int] = HOLDOUT_SEEDS,
    scenario_modes: Iterable[str] = SCENARIO_MODES,
    config: dict[str, Any] | str | Path | None = None,
    domain_randomization: bool = True,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a policy across the full seed x scenario_mode grid.

    Returns
    -------
    per_episode : pd.DataFrame
        One row per (seed, scenario_mode) combination.
    summary : pd.DataFrame
        Aggregated by scenario_mode: mean/std of total episode cost, service
        level, cost breakdown, and average policy call latency.
    """
    student_config = _load_config(config)
    rows: list[dict[str, Any]] = []
    seed_list = list(seeds)
    for scenario_mode in scenario_modes:
        env = IndustrialInventoryEnv(
            student_config=student_config,
            scenario_mode=scenario_mode,
            domain_randomization=domain_randomization,
        )
        for seed in seed_list:
            row = _run_episode(env, policy, seed)
            row["scenario_mode"] = scenario_mode
            rows.append(row)
            if progress:
                print(f"{scenario_mode} seed={seed} cost={row['episode_cost']:.1f}")
    per_episode = pd.DataFrame(rows)
    summary = (
        per_episode.groupby("scenario_mode")
        .agg(
            mean_cost=("episode_cost", "mean"),
            std_cost=("episode_cost", "std"),
            p50_cost=("episode_cost", "median"),
            p90_cost=("episode_cost", lambda s: float(np.percentile(s, 90))),
            mean_service_level=("service_level", "mean"),
            mean_holding=("holding_cost", "mean"),
            mean_stockout=("stockout_cost", "mean"),
            mean_ordering=("ordering_cost", "mean"),
            mean_discarding=("discarding_cost", "mean"),
            mean_call_ms=("avg_policy_call_ms", "mean"),
        )
        .reset_index()
    )
    return per_episode, summary


def summarise_overall(per_episode: pd.DataFrame) -> dict[str, float]:
    """Return a single-line summary averaged across all evaluated episodes."""
    return {
        "mean_cost": float(per_episode["episode_cost"].mean()),
        "std_cost": float(per_episode["episode_cost"].std()),
        "median_cost": float(per_episode["episode_cost"].median()),
        "mean_service_level": float(per_episode["service_level"].mean()),
        "mean_call_ms": float(per_episode["avg_policy_call_ms"].mean()),
        "n_episodes": int(len(per_episode)),
    }


__all__ = [
    "FAST_EVAL_SEEDS",
    "HOLDOUT_SEEDS",
    "SCENARIO_MODES",
    "evaluate_policy",
    "summarise_overall",
]