"""Train A2C on the 99-D Representation C environment."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C

from training_pipelines.training_utils.rep_c_env import build_rep_c_vec_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "training_pipelines" / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train A2C with Representation C")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default="a2c_rep_c")
    parser.add_argument("--no-shaping", action="store_true")
    args = parser.parse_args()

    if args.timesteps < args.n_envs * args.n_steps:
        raise ValueError("timesteps must cover at least one complete rollout")
    if args.n_envs < 1 or args.n_steps < 1:
        raise ValueError("n-envs and n-steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be > 0")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae-lambda must be in [0, 1]")
    if args.ent_coef < 0 or args.vf_coef < 0:
        raise ValueError("ent-coef and vf-coef must be non-negative")

    save_dir = MODELS_DIR / "a2c_rep_c_experiments" / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_rep_c_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        flatten_action=False,
        shaping=not args.no_shaping,
        aggregate_anneal_steps=args.timesteps if not args.no_shaping else None,
    )

    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=args.timesteps)
        model.save(str(save_dir / "final_model"))
    finally:
        train_env.close()

    print(f"A2C Rep-C training done. Run directory: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
