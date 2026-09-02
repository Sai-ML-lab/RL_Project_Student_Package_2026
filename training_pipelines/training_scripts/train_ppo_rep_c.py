"""PPO + Representation C experiment runner.

This is isolated from the frozen PPO champion.  It fixes reward-shaping
annealing for vectorized environments by converting the aggregate timestep
budget to local per-environment steps, while keeping all official evaluation
unshaped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from training_pipelines.src.features.representation_c import REPRESENTATION_C_DIM
from training_pipelines.training_utils.rep_c_env import build_rep_c_vec_env
from training_pipelines.training_scripts.common import MODELS_DIR


DEFAULT_SEED = 20260902


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.timesteps < args.n_envs * args.n_steps:
        raise ValueError("timesteps must cover at least one complete rollout")

    run_name = args.run_name or (
        f"ppo_rep_c_{args.timesteps // 1000}k_"
        f"lr{args.learning_rate:.0e}_g{args.gamma}_seed{args.seed}"
    )
    save_dir = MODELS_DIR / "ppo_rep_c_experiments" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_rep_c_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        flatten_action=False,
        shaping=True,
        aggregate_anneal_steps=args.timesteps,
    )
    eval_env = build_rep_c_vec_env(
        n_envs=1,
        seed=900,
        flatten_action=False,
        shaping=False,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    print(
        "PPO Rep-C configuration: "
        f"dim={REPRESENTATION_C_DIM}, timesteps={args.timesteps}, "
        f"envs={args.n_envs}, rollout={args.n_steps}, batch={args.batch_size}, "
        f"epochs={args.n_epochs}, lr={args.learning_rate}, gamma={args.gamma}, "
        f"gae={args.gae_lambda}, entropy={args.entropy_coef}",
        flush=True,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(save_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    print(f"Saved PPO Rep-C run to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
