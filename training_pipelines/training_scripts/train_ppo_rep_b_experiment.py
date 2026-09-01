"""Isolated PPO experiment using Representation B.

Representation B:
    38 raw normalized observation features
    +
    38 engineered inventory-control features
    =
    76-dimensional policy input.

This experiment is intentionally isolated from the existing PPO champion.

Training:
    IndustrialInventoryEnv
        -> ShapedReward
        -> EngineeredObsWrapper
        -> PPO

Evaluation:
    official unshaped environment
        -> Representation B computed from raw observation
        -> deterministic PPO inference
        -> official episode cost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

for path in (REPO_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from evaluation import evaluate_policy, summarise_overall
from industrial_inventory_env import IndustrialInventoryEnv
from src.features.engineered import (
    EngineeredObsWrapper,
    REPRESENTATION_B_DIM,
    flatten_observation_representation_b,
)
from training_utils.reward_shaping import ShapedReward


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "ppo_rep_b_experiments"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "eval_results"
    / "ppo_rep_b_experiments"
)

DEFAULT_TIMESTEPS = 1_500_000
DEFAULT_SEED = 20260727
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_ENTROPY_COEF = 0.0

N_ENVS = 8
N_STEPS = 512
BATCH_SIZE = 512
N_EPOCHS = 10

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

SCREEN_EVAL_SEEDS = tuple(range(900, 910))

SCENARIO_MODES = (
    "stationary",
    "seasonal",
    "trend",
    "shock",
    "random",
)


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def _load_assigned_config() -> dict[str, Any]:
    """Load the official assigned configuration."""

    config_path = (
        PROJECT_ROOT
        / "assigned_config.json"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def build_rep_b_training_env(
    *,
    anneal_steps: int,
):
    """Build one Representation-B training environment.

    Reward shaping is applied only during training.
    """

    config = _load_assigned_config()

    env = IndustrialInventoryEnv(
        student_config=config,
        scenario_mode="random",
        domain_randomization=True,
    )

    env = ShapedReward(
        env,
        anneal_steps=anneal_steps,
    )

    env = EngineeredObsWrapper(
        env
    )

    return env


def make_vectorized_training_env(
    *,
    n_envs: int,
    seed: int,
    anneal_steps: int,
):
    """Build a DummyVecEnv of Representation-B environments."""

    from stable_baselines3.common.vec_env import DummyVecEnv

    env_fns = []

    for _ in range(n_envs):

        def make_env():
            return build_rep_b_training_env(
                anneal_steps=anneal_steps,
            )

        env_fns.append(make_env)

    vec_env = DummyVecEnv(
        env_fns
    )

    # Seeds are applied by the vectorized environment on reset.
    vec_env.seed(seed)

    return vec_env


# ---------------------------------------------------------------------------
# Official evaluation
# ---------------------------------------------------------------------------


def evaluate_trained_model(
    model: PPO,
) -> dict[str, Any]:
    """Evaluate using the official unshaped cost metric."""

    def policy(observation):
        features = (
            flatten_observation_representation_b(
                observation
            )
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(-1)

        if features.shape != (
            REPRESENTATION_B_DIM,
        ):
            raise ValueError(
                "Representation B produced unexpected "
                f"shape {features.shape}; expected "
                f"{(REPRESENTATION_B_DIM,)}"
            )

        action, _ = model.predict(
            features,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action.shape != (3,):
            raise ValueError(
                "PPO returned unexpected action shape: "
                f"{action.shape}"
            )

        if np.any(action < 0) or np.any(action > 10):
            raise ValueError(
                "PPO returned invalid action indices: "
                f"{action.tolist()}"
            )

        quantities = (
            action * 10
        ).tolist()

        if not all(
            int(q) in range(0, 101, 10)
            for q in quantities
        ):
            raise ValueError(
                f"Invalid quantities: {quantities}"
            )

        return [
            int(q)
            for q in quantities
        ]

    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=SCREEN_EVAL_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=PROJECT_ROOT / "assigned_config.json",
        domain_randomization=True,
        progress=False,
    )

    overall = summarise_overall(
        per_episode
    )

    return {
        "mean_cost": float(
            overall["mean_cost"]
        ),
        "std_cost": float(
            overall["std_cost"]
        ),
        "median_cost": float(
            overall["median_cost"]
        ),
        "mean_service_level": float(
            overall["mean_service_level"]
        ),
        "scenario_summary": scenario_summary.to_dict(
            orient="records"
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_run(
    *,
    timesteps: int,
    seed: int,
    learning_rate: float,
    entropy_coef: float,
    save_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train one isolated PPO + Representation-B experiment."""

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== PPO + Representation B ===",
        flush=True,
    )

    print(
        f"representation_dim={REPRESENTATION_B_DIM} "
        f"timesteps={timesteps:,} "
        f"n_envs={N_ENVS} "
        f"n_steps={N_STEPS} "
        f"lr={learning_rate} "
        f"entropy={entropy_coef} "
        f"gamma={GAMMA} "
        f"gae_lambda={GAE_LAMBDA} "
        f"device={device}",
        flush=True,
    )

    train_env = make_vectorized_training_env(
        n_envs=N_ENVS,
        seed=seed,
        anneal_steps=timesteps,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=entropy_coef,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_advantage=True,
        policy_kwargs={
            "net_arch": [128, 128],
        },
        seed=seed,
        verbose=1,
        device=device,
    )

    try:
        model.learn(
            total_timesteps=timesteps,
        )

        final_model_path = (
            save_dir
            / "final_model"
        )

        model.save(
            str(final_model_path)
        )

        print(
            "\nTraining complete. "
            "Running official local holdout evaluation...",
            flush=True,
        )

        evaluation = evaluate_trained_model(
            model
        )

    finally:
        train_env.close()

    result: dict[str, Any] = {
        "technique": (
            "PPO + Representation B"
        ),
        "representation": "B",
        "observation_dim": (
            REPRESENTATION_B_DIM
        ),
        "timesteps": int(
            timesteps
        ),
        "seed": int(seed),
        "learning_rate": float(
            learning_rate
        ),
        "entropy_coef": float(
            entropy_coef
        ),
        "device": device,
        "n_envs": int(
            N_ENVS
        ),
        "n_steps": int(
            N_STEPS
        ),
        "batch_size": int(
            BATCH_SIZE
        ),
        "n_epochs": int(
            N_EPOCHS
        ),
        "gamma": float(
            GAMMA
        ),
        "gae_lambda": float(
            GAE_LAMBDA
        ),
        "clip_range": float(
            CLIP_RANGE
        ),
        "vf_coef": float(
            VF_COEF
        ),
        "max_grad_norm": float(
            MAX_GRAD_NORM
        ),
        "mean_cost": evaluation["mean_cost"],
        "std_cost": evaluation["std_cost"],
        "median_cost": evaluation["median_cost"],
        "mean_service_level": (
            evaluation["mean_service_level"]
        ),
        "scenario_summary": (
            evaluation["scenario_summary"]
        ),
        "run_dir": str(
            save_dir
        ),
        "final_model": str(
            save_dir
            / "final_model.zip"
        ),
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lr_tag = (
        f"{learning_rate:.0e}"
        .replace("+0", "")
        .replace("+", "")
    )

    result_path = (
        RESULTS_DIR
        / (
            f"ppo_rep_b_"
            f"{timesteps // 1000}k_"
            f"lr{lr_tag}_"
            f"seed{seed}.json"
        )
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== PPO Representation B experiment complete ===",
        flush=True,
    )

    print(
        f"Run directory : {save_dir}",
        flush=True,
    )

    print(
        f"Final model   : "
        f"{save_dir / 'final_model.zip'}",
        flush=True,
    )

    print(
        f"Mean cost     : "
        f"{evaluation['mean_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Std cost      : "
        f"{evaluation['std_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Median cost   : "
        f"{evaluation['median_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Service       : "
        f"{evaluation['mean_service_level']:.6f}",
        flush=True,
    )

    print(
        f"Result file   : {result_path}",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO using the fixed "
            "76-dimensional Representation B."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=DEFAULT_ENTROPY_COEF,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "mps"),
        help="Torch device used by PPO.",
    )

    args = parser.parse_args()

    if args.timesteps < N_STEPS * N_ENVS:
        raise ValueError(
            "For PPO with "
            f"{N_ENVS} environments and "
            f"{N_STEPS}-step rollouts, "
            f"--timesteps must be at least "
            f"{N_ENVS * N_STEPS}."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    if args.entropy_coef < 0:
        raise ValueError(
            "--entropy-coef must be >= 0."
        )

    lr_tag = (
        f"{args.learning_rate:.0e}"
        .replace("+0", "")
        .replace("+", "")
    )

    run_name = (
        args.run_name
        if args.run_name
        else (
            f"ppo_rep_b_"
            f"{args.timesteps // 1000}k_"
            f"lr{lr_tag}"
        )
    )

    save_dir = (
        MODELS_DIR
        / f"{run_name}_seed{args.seed}"
    )

    train_one_run(
        timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        save_dir=save_dir,
        device=args.device,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )