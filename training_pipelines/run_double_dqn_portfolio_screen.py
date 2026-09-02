"""Portfolio-aware Double DQN screening on the standard 35-D observation.

Rep-C DDQN was rejected, so this screen intentionally returns to the standard
FlattenObs representation used by the protected DQN-family baselines and
searches around the retained Double DQN regime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN

from training_scripts.common import EvalCallback, build_eval_vec_env, build_training_vec_env

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "double_dqn_portfolio_screen"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "double_dqn_portfolio_screen"

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 20260904

CONFIGS = {
    "d1_lr2e-4_g0.99_t1000_xf0.25": dict(learning_rate=2e-4, gamma=0.99, target_update_interval=1000, exploration_fraction=0.25),
    "d2_lr3e-4_g0.99_t2000_xf0.25": dict(learning_rate=3e-4, gamma=0.99, target_update_interval=2000, exploration_fraction=0.25),
    "d3_lr5e-4_g0.99_t4000_xf0.25": dict(learning_rate=5e-4, gamma=0.99, target_update_interval=4000, exploration_fraction=0.25),
    "d4_lr5e-4_g0.98_t2000_xf0.40": dict(learning_rate=5e-4, gamma=0.98, target_update_interval=2000, exploration_fraction=0.40),
    "d5_lr7e-4_g0.98_t4000_xf0.30": dict(learning_rate=7e-4, gamma=0.98, target_update_interval=4000, exploration_fraction=0.30),
    "d6_lr3e-4_g0.995_t2000_xf0.30": dict(learning_rate=3e-4, gamma=0.995, target_update_interval=2000, exploration_fraction=0.30),
}


class DoubleDQN(DQN):
    """DQN with the Double-DQN target: online argmax, target evaluation."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, 1, replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def run_one(config_id: str, params: dict, timesteps: int, seed: int) -> dict:
    run_dir = MODELS_DIR / f"{config_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_env = build_training_vec_env(
        n_envs=1,
        flatten_action=True,
        shaping=True,
        seed=seed,
        shaping_kwargs={"anneal_steps": timesteps},
    )
    eval_env = build_eval_vec_env(n_envs=1, flatten_action=True, seed=99)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    model = DoubleDQN(
        "MlpPolicy",
        train_env,
        learning_rate=params["learning_rate"],
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=1.0,
        gamma=params["gamma"],
        train_freq=4,
        gradient_steps=1,
        target_update_interval=params["target_update_interval"],
        exploration_fraction=params["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(str(run_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    best_cost = float("inf")
    best_service = 0.0
    best_step = 0
    # EvalCallback writes evaluations.npz; recover its best deterministic mean.
    npz_path = run_dir / "eval_logs" / "evaluations.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        rewards = np.asarray(data["results"], dtype=np.float64).mean(axis=1)
        timesteps_logged = np.asarray(data["timesteps"], dtype=int)
        idx = int(np.argmin(rewards))
        best_cost = -float(rewards[idx]) * 100.0
        best_step = int(timesteps_logged[idx])
    return {"config_id": config_id, "seed": seed, "timesteps": timesteps, **params, "best_screen_cost": best_cost, "best_screen_step": best_step, "run_dir": str(run_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-configs", type=int, default=len(CONFIGS))
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(CONFIGS.items())[: max(1, min(args.max_configs, len(CONFIGS)))]
    results = []
    print(f"Running {len(selected)} Double DQN configs at {args.timesteps:,} timesteps, seed={args.seed}.", flush=True)
    for config_id, params in selected:
        print(f"\n=== {config_id} ===\n{params}", flush=True)
        row = run_one(config_id, params, args.timesteps, args.seed)
        results.append(row)
        print(f"BEST screen cost={row['best_screen_cost']:,.1f} step={row['best_screen_step']:,}", flush=True)
    results.sort(key=lambda r: r["best_screen_cost"])
    output = RESULTS_DIR / "screening_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== Double DQN portfolio screening ranking ===")
    for rank, row in enumerate(results, 1):
        print(f"{rank:2d}. {row['config_id']} cost={row['best_screen_cost']:,.2f} step={row['best_screen_step']:,}")
    print(f"\nSaved screening results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
