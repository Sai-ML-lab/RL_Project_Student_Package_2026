
"""Shared training utilities for the IITM RL inventory-control project.

This package intentionally has no side effects on import so that the code inside
each submitted `policy.py` can copy the small preprocessing functions inline
(the evaluator sandbox will not have this package available).
"""
from .obs_wrapper import FEATURE_DIM, FlattenObs, flatten_observation
from .action_wrapper import ACTION_SPACE_SIZE, FlattenAction, joint_index_to_quantities
from .reward_shaping import ShapedReward
from .env_factory import load_assigned_config, make_eval_env, make_training_env

__all__ = [
    "ACTION_SPACE_SIZE",
    "FEATURE_DIM",
    "FlattenAction",
    "FlattenObs",
    "ShapedReward",
    "flatten_observation",
    "joint_index_to_quantities",
    "load_assigned_config",
    "make_eval_env",
    "make_training_env",
]