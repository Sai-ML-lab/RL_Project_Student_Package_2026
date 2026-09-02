"""Targeted Double DQN experiment.

This implementation subclasses Stable-Baselines3 DQN and changes only the
target calculation to the Double DQN formulation:

    a* = argmax_a Q_online(s', a)
    target = r + gamma * Q_target(s', a*)

All other DQN settings remain identical to the current successful DQN setup.

The experiment is isolated from submissions/dqn/ so the official DQN artifact
cannot be overwritten accidentally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN

from training_scripts.common import (
    EvalCallback,
    MODELS_DIR,
    build_eval_vec_env,
    build_training_vec_env,
)


class DoubleDQN(DQN):
    """Stable-Baselines3 DQN with Double DQN target computation."""

    def train(
        self,
        gradient_steps: int,
        batch_size: int = 100,
    ) -> None:
        # Switch policy to training mode.
        self.policy.set_training_mode(True)

        # Update learning rate schedule.
        self._update_learning_rate(
            self.policy.optimizer
        )

        losses = []

        for _ in range(gradient_steps):
            # Sample replay buffer.
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )

            # For n-step replay, SB3 supplies gamma**n when appropriate.
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # ----------------------------------------------------------
                # Double DQN:
                #
                # 1. ONLINE network chooses the next action.
                # 2. TARGET network evaluates that chosen action.
                # ----------------------------------------------------------

                next_q_online = self.q_net(
                    replay_data.next_observations
                )

                next_actions = next_q_online.argmax(
                    dim=1,
                    keepdim=True,
                )

                next_q_target = self.q_net_target(
                    replay_data.next_observations
                )

                next_q_values = next_q_target.gather(
                    dim=1,
                    index=next_actions,
                )

                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones)
                    * discounts
                    * next_q_values
                )

            # Current Q estimates from online network.
            current_q_values = self.q_net(
                replay_data.observations
            )

            current_q_values = th.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )

            # Same Huber loss as SB3 DQN.
            loss = F.smooth_l1_loss(
                current_q_values,
                target_q_values,
            )

            losses.append(
                float(loss.item())
            )

            self.policy.optimizer.zero_grad()

            loss.backward()

            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )

            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )

        self.logger.record(
            "train/loss",
            np.mean(losses),
        )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = (
    MODELS_DIR / "double_dqn_experiments"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a targeted Double DQN experiment."
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

    # DQN is a single-environment algorithm.
    train_env = build_training_vec_env(
        n_envs=1,
        flatten_action=True,
        shaping=True,
        seed=args.seed,
        shaping_kwargs={
            "anneal_steps": args.timesteps,
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

    model = DoubleDQN(
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
            "net_arch": [256, 256],
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
        "\nDouble DQN experiment complete."
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