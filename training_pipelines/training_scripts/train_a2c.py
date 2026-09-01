
"""Train an A2C agent on the shaped training env.

Deliberately kept methodologically distinct from PPO: no clipping, no multi-epoch
policy reuse, short n_steps rollouts, GAE lambda=1.0 (undiscounted advantage).
"""
from __future__ import annotations

import argparse

from stable_baselines3 import A2C

from training_scripts.common import (
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
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    args = parser.parse_args()

    save_dir = MODELS_DIR / "a2c"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_training_vec_env(
        n_envs=args.n_envs,
        flatten_action=False,
        shaping=True,
        seed=args.seed,
        shaping_kwargs={"anneal_steps": args.timesteps},
    )
    eval_env = build_eval_vec_env(n_envs=1, flatten_action=False, seed=99)

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
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.98,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(str(save_dir / "final_model"))
    print(f"A2C training done. Best model at: {save_dir / 'best_model.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())