"""Train a DQN agent on the shaped training environment.

The environment uses a flattened Discrete(1331) joint action space.

Each run is saved into a separate experiment directory so that existing
submitted-model artifacts are never overwritten.
"""

from __future__ import annotations

import argparse

from stable_baselines3 import DQN

from training_scripts.common import (
    EvalCallback,
    MODELS_DIR,
    build_eval_vec_env,
    build_training_vec_env,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train DQN for inventory control"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=150_000,
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
        help=(
            "Optional experiment name. If omitted, a name based on "
            "the timestep budget is used."
        ),
    )

    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError("--timesteps must be >= 1.")

    if args.eval_freq < 1:
        raise ValueError("--eval-freq must be >= 1.")

    if args.n_eval_episodes < 1:
        raise ValueError(
            "--n-eval-episodes must be >= 1."
        )

    run_name = (
        args.run_name
        if args.run_name
        else f"{args.timesteps // 1000}k"
    )

    # IMPORTANT:
    # Never overwrite the existing canonical DQN model directory.
    save_dir = (
        MODELS_DIR
        / "dqn_experiments"
        / f"{run_name}_seed{args.seed}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # DQN is a single-environment algorithm.
    train_env = build_training_vec_env(
        n_envs=1,
        flatten_action=True,
        shaping=True,
        seed=args.seed,
        shaping_kwargs={
            "anneal_steps": args.timesteps
        },
    )

    eval_env = build_eval_vec_env(
        n_envs=1,
        flatten_action=True,
        seed=99,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(
            save_dir / "eval_logs"
        ),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=1.0,
        gamma=0.98,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        policy_kwargs={
            "net_arch": [256, 256]
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
        "\nDQN training complete."
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