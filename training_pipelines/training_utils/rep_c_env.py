"""Isolated environment factory for Representation C experiments."""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from industrial_inventory_env import IndustrialInventoryEnv
from training_pipelines.src.features.representation_c import RepresentationCObsWrapper
from training_pipelines.training_utils.action_wrapper import FlattenAction
from training_pipelines.training_utils.reward_shaping import ShapedReward
from training_pipelines.training_utils.env_factory import load_assigned_config


def make_rep_c_env(
    *,
    flatten_action: bool,
    shaping: bool,
    seed: int,
    shaping_kwargs: dict[str, Any] | None = None,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
) -> gym.Env:
    config = load_assigned_config()
    env: gym.Env = IndustrialInventoryEnv(
        student_config=config,
        scenario_mode=scenario_mode,
        domain_randomization=domain_randomization,
    )
    if shaping:
        env = ShapedReward(env, **(shaping_kwargs or {}))
    env = RepresentationCObsWrapper(env)
    if flatten_action:
        env = FlattenAction(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def build_rep_c_vec_env(
    *,
    n_envs: int,
    seed: int,
    flatten_action: bool,
    shaping: bool,
    aggregate_anneal_steps: int | None = None,
) -> DummyVecEnv:
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    shaping_kwargs: dict[str, Any] = {}
    if shaping and aggregate_anneal_steps is not None:
        shaping_kwargs["anneal_steps"] = max(
            1, int(np.ceil(aggregate_anneal_steps / n_envs))
        )

    def factory(rank: int) -> Callable[[], gym.Env]:
        def make() -> gym.Env:
            return make_rep_c_env(
                flatten_action=flatten_action,
                shaping=shaping,
                seed=seed + rank * 1000,
                shaping_kwargs=shaping_kwargs,
            )

        return make

    return DummyVecEnv([factory(i) for i in range(n_envs)])
