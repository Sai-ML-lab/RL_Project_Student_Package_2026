"""Portfolio-aware Double DQN screening using the proven 38-D feature representation.

This intentionally does NOT use Representation C: the earlier DDQN Rep-C
experiment regressed badly. The feature vector here matches the retained DDQN
submission's inference representation exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from industrial_inventory_env import IndustrialInventoryEnv
from training_utils.reward_shaping import ShapedReward
from src.environment.wrappers import JointActionWrapper
from gymnasium import ObservationWrapper, spaces

TRAINING_ROOT = Path(__file__).resolve().parent
MODELS_DIR = TRAINING_ROOT / "models" / "ddqn_portfolio_screen"
RESULTS_DIR = TRAINING_ROOT / "eval_results" / "ddqn_portfolio_screen"
DEFAULT_TIMESTEPS = 150_000
DEFAULT_SEED = 20260904

PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
CAPACITY = 1000.0
REF_DEMAND_MEANS = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)

CONFIGS = {
    "d1_lr3e-4_g0.98_t2000_ef0.02": dict(learning_rate=3e-4, gamma=0.98, target_update_interval=2000, exploration_fraction=0.30, exploration_final_eps=0.02),
    "d2_lr5e-4_g0.98_t2000_ef0.02": dict(learning_rate=5e-4, gamma=0.98, target_update_interval=2000, exploration_fraction=0.30, exploration_final_eps=0.02),
    "d3_lr7e-4_g0.98_t2000_ef0.02": dict(learning_rate=7e-4, gamma=0.98, target_update_interval=2000, exploration_fraction=0.30, exploration_final_eps=0.02),
    "d4_lr5e-4_g0.99_t2000_ef0.02": dict(learning_rate=5e-4, gamma=0.99, target_update_interval=2000, exploration_fraction=0.30, exploration_final_eps=0.02),
    "d5_lr5e-4_g0.98_t4000_ef0.02": dict(learning_rate=5e-4, gamma=0.98, target_update_interval=4000, exploration_fraction=0.30, exploration_final_eps=0.02),
    "d6_lr5e-4_g0.98_t2000_ef0.01": dict(learning_rate=5e-4, gamma=0.98, target_update_interval=2000, exploration_fraction=0.35, exploration_final_eps=0.01),
}


def flatten_ddqn_observation(observation) -> np.ndarray:
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32)
    day = float(np.asarray(observation["day"]).reshape(-1)[0])
    capacity_utilisation = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])

    last3_mean = demand_history[-3:].mean(axis=0)
    last7_mean = demand_history.mean(axis=0)
    last7_std = demand_history.std(axis=0)
    inv_position_units = inventory + pipeline.sum(axis=1)
    inv_position_volume = inv_position_units * PRODUCT_VOLUMES

    features = np.concatenate(
        [
            inventory / 100.0,
            pipeline.reshape(-1) / 100.0,
            last3_mean / REF_DEMAND_MEANS,
            last7_mean / REF_DEMAND_MEANS,
            last7_std / REF_DEMAND_MEANS,
            np.asarray([day / 50.0], dtype=np.float32),
            np.asarray([capacity_utilisation], dtype=np.float32),
            inv_position_units / 200.0,
            inv_position_volume / CAPACITY,
            (inv_position_units - last3_mean) / 100.0,
        ],
        dtype=np.float32,
    )
    if features.shape != (38,):
        raise RuntimeError(f"DDQN feature shape mismatch: {features.shape}; expected (38,)")
    return features


class DDQNObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(38,), dtype=np.float32)

    def observation(self, observation):
        return flatten_ddqn_observation(observation)


class DoubleDQN(DQN):
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
        self.logger.record("train/loss", float(np.mean(losses)))


def _make_env(*, shaping: bool, timesteps: int, config: dict):
    env = IndustrialInventoryEnv(student_config=config, scenario_mode="random", domain_randomization=True)
    if shaping:
        env = ShapedReward(env, anneal_steps=timesteps)
    env = DDQNObsWrapper(env)
    env = JointActionWrapper(env)
    return Monitor(env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-configs", type=int, default=len(CONFIGS))
    args = parser.parse_args()

    with open(TRAINING_ROOT / "assigned_config.json", "r", encoding="utf-8") as f:
        assigned_config = json.load(f)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(CONFIGS.items())[: max(1, min(args.max_configs, len(CONFIGS)))]
    rows = []

    for config_id, params in selected:
        save_dir = MODELS_DIR / f"{config_id}_seed{args.seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {config_id} ===\n{params}", flush=True)

        train_env = DummyVecEnv([lambda: _make_env(shaping=True, timesteps=args.timesteps, config=assigned_config)])
        eval_env = DummyVecEnv([lambda: _make_env(shaping=False, timesteps=args.timesteps, config=assigned_config)])
        callback = EvalCallback(eval_env, best_model_save_path=str(save_dir), log_path=str(save_dir / "eval_logs"), eval_freq=10_000, n_eval_episodes=10, deterministic=True, verbose=0)

        model = DoubleDQN(
            "MlpPolicy", train_env,
            learning_rate=params["learning_rate"],
            buffer_size=200_000, learning_starts=5_000, batch_size=256,
            tau=1.0, gamma=params["gamma"], train_freq=4, gradient_steps=1,
            target_update_interval=params["target_update_interval"],
            exploration_fraction=params["exploration_fraction"],
            exploration_initial_eps=1.0,
            exploration_final_eps=params["exploration_final_eps"],
            max_grad_norm=10.0,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed, verbose=0, device="cpu",
        )
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback)
            model.save(str(save_dir / "final_model"))
        finally:
            train_env.close(); eval_env.close()

        best_path = save_dir / "best_model.zip"
        if not best_path.exists():
            best_path = save_dir / "final_model.zip"
        eval_env_single = _make_env(shaping=False, timesteps=args.timesteps, config=assigned_config)
        model_eval = DoubleDQN.load(str(best_path), env=eval_env_single, device="cpu")
        costs, services = [], []
        for s in range(900, 910):
            obs, _ = eval_env_single.reset(seed=s)
            done = False; cost = 0.0; demand = fulfilled = 0
            while not done:
                action, _ = model_eval.predict(obs, deterministic=True)
                obs, _, term, trunc, info = eval_env_single.step(action)
                done = bool(term or trunc)
                cost += float(info["costs"]["daily_total"])
                demand += int(np.sum(info["demand"]))
                fulfilled += int(np.sum(info["fulfilled_demand"]))
            costs.append(cost); services.append(fulfilled / max(demand, 1))
        eval_env_single.close()
        row = {"config_id": config_id, **params, "seed": args.seed, "best_eval_cost": float(np.mean(costs)), "best_eval_service": float(np.mean(services)), "checkpoint": str(best_path)}
        rows.append(row)
        print(f"SCREEN cost={row['best_eval_cost']:,.2f} service={row['best_eval_service']:.6f}", flush=True)

    rows.sort(key=lambda r: r["best_eval_cost"])
    output = RESULTS_DIR / "screening_results.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== Double DQN portfolio screening ranking ===")
    for i, row in enumerate(rows, 1):
        print(f"{i:2d}. {row['config_id']} cost={row['best_eval_cost']:,.2f} service={row['best_eval_service']:.6f}")
    print(f"\nSaved screening results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
