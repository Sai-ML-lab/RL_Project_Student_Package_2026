"""Shared helpers for Representation-B RL experiments.

This module centralizes the environment construction and the authoritative
200-episode local holdout evaluation used by the candidate experiments.

Representation B is applied after reward shaping during training:
    IndustrialInventoryEnv -> ShapedReward -> EngineeredObsWrapper

Evaluation is always unshaped and uses the root official evaluation harness.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

for _path in (REPO_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from industrial_inventory_env import IndustrialInventoryEnv
from training_utils.env_factory import load_assigned_config
from training_utils.reward_shaping import ShapedReward
from src.environment.action_codec import joint_index_to_quantities
from src.features.engineered import (
    EngineeredObsWrapper,
    REPRESENTATION_B_DIM,
    flatten_observation_representation_b,
)

__all__ = [
    "PROJECT_ROOT",
    "REPO_ROOT",
    "REPRESENTATION_B_DIM",
    "build_rep_b_vec_env",
    "evaluate_sb3_multidiscrete",
    "evaluate_joint_q_model",
    "make_rep_b_env",
    "load_assigned_config",
]


def make_rep_b_env(
    *,
    training: bool,
    seed: int | None = None,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
    shaping_kwargs: dict | None = None,
) -> gym.Env:
    """Build an environment with Representation B and MultiDiscrete actions."""
    config = load_assigned_config(PROJECT_ROOT / "assigned_config.json")

    env: gym.Env = IndustrialInventoryEnv(
        student_config=config,
        scenario_mode=scenario_mode,
        domain_randomization=domain_randomization,
    )

    if training:
        env = ShapedReward(
            env,
            **(shaping_kwargs or {}),
        )

    # Do NOT use JointActionWrapper here: A2C/A3C use the native
    # MultiDiscrete([11, 11, 11]) action space.
    env = EngineeredObsWrapper(env)

    if seed is not None:
        env.reset(seed=int(seed))

    return env


def build_rep_b_vec_env(
    *,
    n_envs: int,
    seed: int,
    aggregate_timesteps: int,
    shaping: bool = True,
) -> DummyVecEnv:
    """Build a DummyVecEnv for A2C-style algorithms.

    Reward-shaping annealing is expressed in local environment transitions,
    therefore an aggregate transition budget is divided across workers.
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if aggregate_timesteps < 1:
        raise ValueError("aggregate_timesteps must be >= 1")

    local_anneal_steps = max(
        1,
        int(np.ceil(aggregate_timesteps / n_envs)),
    )

    def make_one(rank: int) -> Callable[[], gym.Env]:
        def factory() -> gym.Env:
            env = make_rep_b_env(
                training=shaping,
                seed=seed + 1000 * rank,
                shaping_kwargs=(
                    {"anneal_steps": local_anneal_steps}
                    if shaping
                    else None
                ),
            )
            return Monitor(env)

        return factory

    vec_env = DummyVecEnv(
        [make_one(rank) for rank in range(n_envs)]
    )
    return vec_env


def evaluate_sb3_multidiscrete(model) -> dict:
    """Evaluate an SB3 model on the official 200-episode holdout."""
    def policy(observation):
        features = flatten_observation_representation_b(observation)
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (REPRESENTATION_B_DIM,):
            raise ValueError(
                f"Expected Representation B shape "
                f"({REPRESENTATION_B_DIM},), got {features.shape}"
            )

        action, _ = model.predict(
            features,
            deterministic=True,
        )
        action = np.asarray(action, dtype=np.int64).reshape(-1)

        if action.shape != (3,):
            raise ValueError(
                f"Expected MultiDiscrete action shape (3,), got {action.shape}"
            )

        if np.any(action < 0) or np.any(action > 10):
            raise ValueError(
                f"Invalid action indices: {action.tolist()}"
            )

        return [int(x * 10) for x in action]

    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=HOLDOUT_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=PROJECT_ROOT / "assigned_config.json",
        domain_randomization=True,
        progress=False,
    )
    overall = summarise_overall(per_episode)

    return {
        "mean_cost": overall["mean_cost"],
        "std_cost": overall["std_cost"],
        "median_cost": overall["median_cost"],
        "mean_service_level": overall["mean_service_level"],
        "mean_call_ms": overall["mean_call_ms"],
        "n_episodes": overall["n_episodes"],
        "scenario_summary": scenario_summary.to_dict(orient="records"),
    }


def evaluate_joint_q_model(q_model: Callable[[np.ndarray], int]) -> dict:
    """Evaluate a callable returning a 1331-action joint index."""
    def policy(observation):
        features = flatten_observation_representation_b(observation)
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (REPRESENTATION_B_DIM,):
            raise ValueError(
                f"Expected Representation B shape "
                f"({REPRESENTATION_B_DIM},), got {features.shape}"
            )

        joint_index = int(q_model(features))
        return joint_index_to_quantities(joint_index)

    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=HOLDOUT_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=PROJECT_ROOT / "assigned_config.json",
        domain_randomization=True,
        progress=False,
    )
    overall = summarise_overall(per_episode)

    return {
        "mean_cost": overall["mean_cost"],
        "std_cost": overall["std_cost"],
        "median_cost": overall["median_cost"],
        "mean_service_level": overall["mean_service_level"],
        "mean_call_ms": overall["mean_call_ms"],
        "n_episodes": overall["n_episodes"],
        "scenario_summary": scenario_summary.to_dict(orient="records"),
    }
