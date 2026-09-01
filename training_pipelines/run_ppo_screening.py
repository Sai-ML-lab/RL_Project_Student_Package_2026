"""PPO hyperparameter screening.

Screens a small grid of PPO configurations using the official unshaped
evaluation environment.

The SB3 EvalCallback is retained for training diagnostics, but candidate
selection is based on explicitly computed official episode cost. This avoids
silently accepting an invalid screening result such as -Infinity when no
callback evaluation was recorded.
"""

from __future__ import annotations

import argparse
from email import parser
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make both repository-root modules (e.g. evaluation.py) and the
# training_pipelines package importable when this file is executed directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from evaluation import evaluate_policy, summarise_overall
from training_scripts.common import (
    build_eval_vec_env,
    build_training_vec_env,
)
from training_utils.obs_wrapper import flatten_observation


MODELS_DIR = PROJECT_ROOT / "models" / "ppo_v2"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase6"

ENT_COEFS = (0.0, 0.001, 0.003)
LEARNING_RATES = (1e-4, 3e-4)

SCREEN_TIMESTEPS = 500_000
N_ENVS = 8
SEED = 20260825

# Small, fixed screening set.
SCREEN_EVAL_SEEDS = tuple(range(900, 910))

SCENARIO_MODES = (
    "stationary",
    "seasonal",
    "trend",
    "shock",
    "random",
)


def _config_id(
    ent_coef: float,
    lr: float,
) -> str:
    return f"ent{ent_coef}_lr{lr}"


def _evaluate_trained_model(
    model: PPO,
) -> dict[str, float]:
    """Evaluate a trained PPO model using the official unshaped environment."""

    def policy(observation):
        # The PPO model was trained on the 35-dimensional flattened
        # observation representation.
        features = flatten_observation(observation)

        # PPO predicts MultiDiscrete action indices in {0, ..., 10}.
        action, _ = model.predict(
            features,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action.shape[0] != 3:
            raise ValueError(
                f"PPO returned {action.shape[0]} action values; expected 3."
            )

        if np.any(action < 0) or np.any(action > 10):
            raise ValueError(
                f"PPO returned invalid action indices: {action.tolist()}"
            )

        # Convert environment action indices {0,...,10} into the official
        # leaderboard quantities {0,10,...,100}.
        quantities = (action * 10).tolist()

        return [int(q) for q in quantities]

    per_episode, _summary = evaluate_policy(
        policy,
        seeds=SCREEN_EVAL_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=PROJECT_ROOT / "assigned_config.json",
        domain_randomization=True,
        progress=False,
    )

    overall = summarise_overall(per_episode)

    return {
        "mean_cost": float(
            overall["mean_cost"]
        ),
        "std_cost": float(
            overall["std_cost"]
        ),
        "mean_service_level": float(
            overall["mean_service_level"]
        ),
    }

def train_one_config(
    ent_coef: float,
    lr: float,
    timesteps: int,
    save_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one PPO hyperparameter configuration."""

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_env = build_training_vec_env(
        n_envs=N_ENVS,
        flatten_action=False,
        shaping=True,
        seed=SEED,
        shaping_kwargs={
            "anneal_steps": timesteps
        },
    )

    eval_env = build_eval_vec_env(
        n_envs=1,
        flatten_action=False,
        seed=99,
        scenario_mode="random",
    )

    # Evaluate roughly every 25,000 aggregate transitions during real runs.
    # For a tiny smoke test, reduce the frequency so at least one evaluation
    # can occur before training finishes.
    eval_freq = max(
        min(
            25_000 // N_ENVS,
            timesteps,
        ),
        1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(
            save_dir / "eval_logs"
        ),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs={
            "net_arch": [128, 128]
        },
        seed=SEED,
        verbose=0,
        device="cpu",
    )

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=eval_callback,
        )

        model.save(
            str(save_dir / "final_model")
        )

        # The explicit official evaluation below is the authoritative
        # screening metric. The callback is only auxiliary training
        # diagnostics.
        evaluation = _evaluate_trained_model(
            model
        )

    finally:
        train_env.close()
        eval_env.close()

    callback_reward = (
        float(eval_callback.best_mean_reward)
        if np.isfinite(
            eval_callback.best_mean_reward
        )
        else None
    )

    return {
        "ent_coef": float(ent_coef),
        "learning_rate": float(lr),
        "screen_timesteps": int(timesteps),
        "eval_freq": int(eval_freq),
        "best_callback_reward": callback_reward,
        **evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen PPO hyperparameters "
            "for inventory control."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=SCREEN_TIMESTEPS,
    )

    parser.add_argument(
    "--learning-rates",
    type=float,
    nargs="+",
    default=list(LEARNING_RATES),
    )

    parser.add_argument(
        "--ent-coefs",
        type=float,
        nargs="+",
        default=list(ENT_COEFS),
    )

    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError(
            "--timesteps must be >= 1."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict[str, Any]] = []

    for ent_coef in args.ent_coefs:
        for lr in args.learning_rates:

            config_id = _config_id(
                ent_coef,
                lr,
            )

            save_dir = (
                MODELS_DIR / config_id
            )

            print(
                f"\n=== training {config_id} "
                f"({args.timesteps:,} timesteps) ===",
                flush=True,
            )

            row = train_one_config(
                ent_coef=ent_coef,
                lr=lr,
                timesteps=args.timesteps,
                save_dir=save_dir,
            )

            row["config_id"] = config_id

            results.append(row)

            callback_text = (
                "None"
                if row["best_callback_reward"] is None
                else f"{row['best_callback_reward']:.3f}"
            )

            print(
                f"  mean_cost="
                f"{row['mean_cost']:,.1f} "
                f"std="
                f"{row['std_cost']:,.1f} "
                f"service="
                f"{row['mean_service_level']:.4f} "
                f"callback_reward="
                f"{callback_text}",
                flush=True,
            )

    # Lower official cost is better.
    results.sort(
        key=lambda row: row["mean_cost"]
    )

    output_path = (
        RESULTS_DIR
        / "ppo_screening_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    print(
        "\n=== Ranked PPO screening results "
        "(lowest official cost first) ==="
    )

    for rank, row in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['config_id']} "
            f"cost="
            f"{row['mean_cost']:,.1f} "
            f"std="
            f"{row['std_cost']:,.1f} "
            f"service="
            f"{row['mean_service_level']:.4f}",
            flush=True,
        )

    print(
        f"\nSaved results to: {output_path}"
    )


if __name__ == "__main__":
    main()