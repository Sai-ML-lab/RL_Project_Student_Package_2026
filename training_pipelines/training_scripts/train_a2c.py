
"""Train an A2C agent on the shaped training environment."""
from __future__ import annotations

import argparse

from stable_baselines3 import A2C

from training_pipelines.training_scripts.common import (
    EvalCallback,
    MODELS_DIR,
    build_eval_vec_env,
    build_training_vec_env,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train A2C for inventory control")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="a2c")
    args = parser.parse_args()

    if args.timesteps < args.n_envs * args.n_steps:
        raise ValueError("timesteps must cover at least one complete rollout")
    if args.n_envs < 1 or args.n_steps < 1:
        raise ValueError("n-envs and n-steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be > 0")

    save_dir = MODELS_DIR / "a2c_experiments" / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_training_vec_env(
        n_envs=args.n_envs,
        flatten_action=False,
        shaping=True,
        seed=args.seed,
        shaping_kwargs={"anneal_steps": args.timesteps},
    )
    eval_env = build_eval_vec_env(n_envs=1, flatten_action=False, seed=900)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
        model.save(str(save_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    print(f"A2C training done. Run directory: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
