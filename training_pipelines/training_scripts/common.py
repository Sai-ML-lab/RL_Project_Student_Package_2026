"""Shared helpers for training scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Support both invocation styles:
#   python -m training_pipelines.training_scripts.<script>
# and
#   cd training_pipelines && python training_scripts/<script>.py
try:
    from training_pipelines.training_utils import (
        load_assigned_config,
        make_eval_env,
        make_training_env,
    )
except ModuleNotFoundError:
    from training_utils import (
        load_assigned_config,
        make_eval_env,
        make_training_env,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def build_training_vec_env(
    n_envs: int = 4,
    *,
    flatten_action: bool = False,
    shaping: bool = True,
    seed: int = 20260727,
    shaping_kwargs: dict | None = None,
) -> DummyVecEnv:
    """Build a vectorized training environment."""
    config = load_assigned_config()
    effective_shaping_kwargs = dict(shaping_kwargs or {})

    if shaping and "anneal_steps" in effective_shaping_kwargs:
        total_training_steps = int(effective_shaping_kwargs["anneal_steps"])
        if total_training_steps < 1:
            raise ValueError("anneal_steps must be >= 1.")

        effective_shaping_kwargs["anneal_steps"] = max(
            1,
            int(np.ceil(total_training_steps / max(1, n_envs))),
        )

        print(
            f"[reward shaping] aggregate_steps={total_training_steps:,} "
            f"n_envs={n_envs} "
            f"local_anneal_steps={effective_shaping_kwargs['anneal_steps']:,}",
            flush=True,
        )

    def _factory(rank: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            env = make_training_env(
                config,
                scenario_mode="random",
                domain_randomization=True,
                flatten_action=flatten_action,
                shaping=shaping,
                shaping_kwargs=effective_shaping_kwargs,
            )
            env = Monitor(env)
            env.reset(seed=seed + 1000 * rank)
            return env

        return _make

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


def build_eval_vec_env(
    n_envs: int = 1,
    *,
    flatten_action: bool = False,
    seed: int = 12345,
    scenario_mode: str = "random",
) -> DummyVecEnv:
    config = load_assigned_config()

    def _factory(rank: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            env = make_eval_env(
                config,
                scenario_mode=scenario_mode,
                domain_randomization=True,
                flatten_action=flatten_action,
            )
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _make

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


__all__ = [
    "EvalCallback",
    "LOGS_DIR",
    "MODELS_DIR",
    "build_eval_vec_env",
    "build_training_vec_env",
]
