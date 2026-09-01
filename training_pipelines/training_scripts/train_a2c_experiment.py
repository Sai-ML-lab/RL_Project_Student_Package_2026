"""Targeted A2C experiment for the RL inventory project.

This script preserves the canonical submitted A2C model and stores new
experiments under a separate directory.

Round 1 changes only the training budget; all A2C hyperparameters remain
identical to the current submitted configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C

from training_scripts.common import (
    EvalCallback,
    MODELS_DIR,
    build_eval_vec_env,
    build_training_vec_env,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = (
    MODELS_DIR / "a2c_experiments"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a targeted A2C experiment."
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError(
            "--timesteps must be >= 1."
        )

    if args.eval_freq < 1:
        raise ValueError(
            "--eval-freq must be >= 1."
        )

    if args.n_eval_episodes < 1:
        raise ValueError(
            "--n-eval-episodes must be >= 1."
        )

    run_name = (
        args.run_name
        if args.run_name
        else f"{args.timesteps // 1000}k"
    )

    save_dir = (
        EXPERIMENTS_DIR
        / f"{run_name}_seed{args.seed}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A2C uses four parallel environments in the existing training setup.
    train_env = build_training_vec_env(
        n_envs=4,
        flatten_action=False,
        shaping=True,
        seed=args.seed,
        shaping_kwargs={
            "anneal_steps": args.timesteps,
        },
    )

    eval_env = build_eval_vec_env(
        n_envs=1,
        flatten_action=False,
        seed=99,
    )

    # eval_freq is expressed in vectorized environment steps.
    # Convert the requested aggregate evaluation interval accordingly.
    callback_eval_freq = max(
        args.eval_freq // 4,
        1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(
            save_dir / "eval_logs"
        ),
        eval_freq=callback_eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.98,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs={
            "net_arch": [128, 128]
        },
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
        )

        model.save(
            str(save_dir / "final_model")
        )

    finally:
        train_env.close()
        eval_env.close()

    print(
        "\nA2C experiment complete."
    )
    print(
        f"Run directory : {save_dir}"
    )
    print(
        f"Final model   : "
        f"{save_dir / 'final_model.zip'}"
    )
    print(
        f"Best model    : "
        f"{save_dir / 'best_model.zip'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())