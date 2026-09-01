
"""PPO recovery: screening sweep over entropy coefficient x learning rate
(Phase 6, doc section 9.2). Does NOT touch iteration-1's `train_ppo.py` or
its Phase-0-protected model -- saves to a separate `models/ppo_v2/` tree.

Core settings held constant per doc: n_envs=8, n_steps=512, batch_size=512,
n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
max_grad_norm=0.5, normalize_advantage=True. Screens at 500,000 transitions;
promote the top 2 configs afterwards via run_ppo_full_training.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from training_scripts.common import build_eval_vec_env, build_training_vec_env

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "ppo_v2"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase6"

ENT_COEFS = (0.0, 0.001, 0.003)
LEARNING_RATES = (1e-4, 3e-4)
SCREEN_TIMESTEPS = 500_000
N_ENVS = 8
SEED = 20260825


def _config_id(ent_coef: float, lr: float) -> str:
    return f"ent{ent_coef}_lr{lr}"


def train_one_config(ent_coef: float, lr: float, timesteps: int, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_training_vec_env(
        n_envs=N_ENVS,
        flatten_action=False,
        shaping=True,
        seed=SEED,
        shaping_kwargs={"anneal_steps": timesteps},
    )
    eval_env = build_eval_vec_env(n_envs=1, flatten_action=False, seed=99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=max(25_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
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
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(str(save_dir / "final_model"))

    best_mean_reward = float(eval_callback.best_mean_reward)
    return {"ent_coef": ent_coef, "learning_rate": lr, "best_mean_reward": best_mean_reward}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=SCREEN_TIMESTEPS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ent_coef in ENT_COEFS:
        for lr in LEARNING_RATES:
            config_id = _config_id(ent_coef, lr)
            save_dir = MODELS_DIR / config_id
            print(f"=== training {config_id} ({args.timesteps} timesteps) ===", flush=True)
            row = train_one_config(ent_coef, lr, args.timesteps, save_dir)
            results.append(row)
            print(f"  best_mean_reward={row['best_mean_reward']:.3f}", flush=True)

    results.sort(key=lambda r: r["best_mean_reward"], reverse=True)  # higher reward = lower cost
    with open(RESULTS_DIR / "ppo_screening_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== Ranked screening results (best mean reward first) ===")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()