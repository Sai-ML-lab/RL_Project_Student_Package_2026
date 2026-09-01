
"""Env factory for Phase 4+ custom algorithms: joint action space (1331) +
Representation B features (raw + engineered, 76-dim) per doc section 7.1
("Input: raw plus engineered features").
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from industrial_inventory_env import IndustrialInventoryEnv
from training_utils.env_factory import load_assigned_config
from training_utils.reward_shaping import ShapedReward

from src.environment.wrappers import JointActionWrapper
from src.features.engineered import EngineeredObsWrapper


def make_training_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
    shaping: bool = True,
    shaping_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    if config is None:
        config = load_assigned_config()
    env = IndustrialInventoryEnv(
        student_config=config, scenario_mode=scenario_mode, domain_randomization=domain_randomization
    )
    if shaping:
        env = ShapedReward(env, **(shaping_kwargs or {}))
    env = JointActionWrapper(env)
    env = EngineeredObsWrapper(env)
    return env


def make_eval_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
) -> gym.Env:
    """Never shaped -- evaluation always uses the official reward."""
    if config is None:
        config = load_assigned_config()
    env = IndustrialInventoryEnv(
        student_config=config, scenario_mode=scenario_mode, domain_randomization=domain_randomization
    )
    env = JointActionWrapper(env)
    env = EngineeredObsWrapper(env)
    return env


def make_raw_training_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
    shaping: bool = True,
    shaping_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    """Base env (Dict obs, MultiDiscrete action) + optional reward shaping --
    for algorithms (TD(lambda), Tabular SARSA) that discretize observations
    and decode a reduced action catalogue themselves (Phase 5, section 8)."""
    if config is None:
        config = load_assigned_config()
    env = IndustrialInventoryEnv(
        student_config=config, scenario_mode=scenario_mode, domain_randomization=domain_randomization
    )
    if shaping:
        env = ShapedReward(env, **(shaping_kwargs or {}))
    return env


def make_raw_eval_env(
    config: dict[str, Any] | None = None,
    *,
    scenario_mode: str = "random",
    domain_randomization: bool = True,
) -> gym.Env:
    """Never shaped -- evaluation always uses the official reward."""
    if config is None:
        config = load_assigned_config()
    return IndustrialInventoryEnv(
        student_config=config, scenario_mode=scenario_mode, domain_randomization=domain_randomization
    )
