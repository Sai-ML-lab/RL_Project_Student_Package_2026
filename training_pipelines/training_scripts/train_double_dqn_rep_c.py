"""Double DQN + Representation C experiment runner.

The target uses the online network for action selection and the target network
for action evaluation.  Existing frozen Double DQN artifacts are isolated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from training_pipelines.src.features.representation_c import REPRESENTATION_C_DIM
from training_pipelines.training_utils.rep_c_env import build_rep_c_vec_env
from training_pipelines.training_scripts.common import MODELS_DIR


class DoubleDQN(DQN):
    """Stable-Baselines3 DQN with the Double-DQN target calculation."""

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
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, 1, replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--exploration-fraction", type=float, default=0.30)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or (
        f"ddqn_rep_c_{args.timesteps // 1000}k_"
        f"lr{args.learning_rate:.0e}_g{args.gamma}_seed{args.seed}"
    )
    save_dir = MODELS_DIR / "double_dqn_rep_c_experiments" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_rep_c_vec_env(
        n_envs=1,
        seed=args.seed,
        flatten_action=True,
        shaping=True,
        aggregate_anneal_steps=args.timesteps,
    )
    eval_env = build_rep_c_vec_env(
        n_envs=1,
        seed=900,
        flatten_action=True,
        shaping=False,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = DoubleDQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_frequency,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=10.0,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    print(
        "DDQN Rep-C configuration: "
        f"dim={REPRESENTATION_C_DIM}, timesteps={args.timesteps}, "
        f"lr={args.learning_rate}, gamma={args.gamma}, "
        f"target={args.target_update_interval}, eps={args.exploration_final_eps}",
        flush=True,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(save_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()

    print(f"Saved DDQN Rep-C run to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
