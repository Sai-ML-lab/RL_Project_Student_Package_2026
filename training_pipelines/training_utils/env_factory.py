
"""Env factories used by every training/evaluation script.

`make_training_env` layers optional wrappers in this order:
    IndustrialInventoryEnv -> ShapedReward (optional) -> FlattenObs
      -> FlattenAction (optional, DQN only)

`make_eval_env` never applies reward shaping so evaluation always uses the
official reward per section 11 of the problem statement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym

from industrial_inventory_env import IndustrialInventoryEnv

from .action_wrapper import FlattenAction
from .obs_wrapper import FlattenObs
from .reward_shaping import ShapedReward

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "assigned_config.json"


def load_assigned_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the persisted student configuration written during Phase 0."""
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _base_env(
    config: dict[str, Any],
    scenario_mode: str = "random",
    domain_randomization: bool = True,
) -> gym.Env:
    return IndustrialInventoryEnv(
        student_config=config,
        scenario_mode=scenario_mode,
        domain_randomization=domain_randomization,
    )


def make_training_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
    flatten_action: bool = False,
    shaping: bool = True,
    shaping_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    """Build a wrapped environment suitable for training."""
    if config is None:
        config = load_assigned_config()
    env = _base_env(config, scenario_mode, domain_randomization)
    if shaping:
        env = ShapedReward(env, **(shaping_kwargs or {}))
    env = FlattenObs(env)
    if flatten_action:
        env = FlattenAction(env)
    return env


def make_eval_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
    flatten_action: bool = False,
) -> gym.Env:
    """Build a wrapped environment suitable for evaluation.

    Reward shaping is never applied — evaluation uses the official reward only.
    """
    if config is None:
        config = load_assigned_config()
    env = _base_env(config, scenario_mode, domain_randomization)
    env = FlattenObs(env)
    if flatten_action:
        env = FlattenAction(env)
    return env