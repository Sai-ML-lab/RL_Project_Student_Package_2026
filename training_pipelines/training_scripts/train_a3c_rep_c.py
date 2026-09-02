"""Lightweight true A3C trainer for the inventory-control project.

This implementation uses multiple independent environment workers, a shared
actor-critic network in shared memory, and a shared Adam optimizer. Workers
periodically copy the shared parameters, collect short on-policy rollouts, and
apply gradients to the shared model. The official environment is unchanged.

Actions are represented internally as one categorical variable over all 1331
joint actions, then decoded back to MultiDiscrete([11,11,11]) indices before
env.step().
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training_pipelines.src.environment.action_codec import (
    JOINT_ACTION_SIZE,
    joint_index_to_multidiscrete,
)
from training_pipelines.src.features.representation_c import (
    REPRESENTATION_C_DIM,
    flatten_observation_representation_c,
)
from training_pipelines.training_utils.rep_c_env import make_rep_c_env
from training_pipelines.training_scripts.common import MODELS_DIR


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = REPRESENTATION_C_DIM, n_actions: int = JOINT_ACTION_SIZE) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


class SharedAdam(torch.optim.Adam):
    """Adam with optimizer state moved to shared memory for A3C workers."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker(
    rank: int,
    shared_model: ActorCritic,
    optimizer: SharedAdam,
    counter,
    lock,
    args,
) -> None:
    seed = args.seed + 1009 * rank
    _seed_everything(seed)

    env = make_rep_c_env(
        flatten_action=False,
        shaping=args.shaping,
        seed=seed,
        shaping_kwargs={"anneal_steps": max(1, args.total_steps // max(1, args.workers))},
        scenario_mode="random",
        domain_randomization=True,
    )

    local_model = ActorCritic()
    local_model.load_state_dict(shared_model.state_dict())
    local_model.train()

    obs, _ = env.reset(seed=seed)
    obs = flatten_observation_representation_c(obs)
    episode_return = 0.0
    episode_steps = 0
    started = time.perf_counter()

    while True:
        with counter.get_lock():
            if counter.value >= args.total_steps:
                break
            remaining = args.total_steps - counter.value
            rollout_steps = min(args.rollout_steps, remaining)
            counter.value += rollout_steps

        local_model.load_state_dict(shared_model.state_dict())

        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        dones = []

        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = local_model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            action_idx = int(action.item())
            multi_action = np.asarray(joint_index_to_multidiscrete(action_idx), dtype=np.int64)

            next_obs, reward, terminated, truncated, _info = env.step(multi_action)
            done = bool(terminated or truncated)
            next_obs = flatten_observation_representation_c(next_obs)

            observations.append(obs)
            actions.append(action_idx)
            rewards.append(float(reward))
            values.append(value.squeeze(0))
            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            dones.append(float(done))

            episode_return += float(reward)
            episode_steps += 1
            obs = next_obs

            if done:
                obs_raw, _ = env.reset()
                obs = flatten_observation_representation_c(obs_raw)
                if rank == 0:
                    print(
                        f"[A3C worker {rank}] episode return={episode_return:.2f} steps={episode_steps}",
                        flush=True,
                    )
                episode_return = 0.0
                episode_steps = 0

        with torch.no_grad():
            next_value = torch.zeros(1)
            if not dones[-1]:
                _, next_value_batch = local_model(torch.from_numpy(obs).float().unsqueeze(0))
                next_value = next_value_batch.detach().cpu()

        returns = []
        advantages = []
        gae = torch.tensor(0.0)
        for t in reversed(range(rollout_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * next_value * mask - values[t].detach()
            gae = delta + args.gamma * args.gae_lambda * mask * gae
            advantages.append(gae)
            returns.append(gae + values[t].detach())
            next_value = values[t].detach()

        advantages.reverse()
        returns.reverse()

        advantage_t = torch.stack(advantages)
        return_t = torch.stack(returns)
        value_t = torch.stack(values)
        log_prob_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)

        if args.normalize_advantage:
            advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std(unbiased=False) + 1e-8)

        policy_loss = -(log_prob_t * advantage_t.detach()).mean()
        value_loss = F.smooth_l1_loss(value_t, return_t.detach())
        entropy_bonus = entropy_t.mean()
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad_norm)

        # Copy local gradients into shared parameters, then apply a shared Adam
        # update. The shared optimizer state permits asynchronous worker updates.
        for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                shared_param._grad = local_param.grad.detach().clone()
        optimizer.step()

        if rank == 0 and (counter.value // args.rollout_steps) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[A3C] global_steps={counter.value:,}/{args.total_steps:,} "
                f"fps={counter.value / max(elapsed, 1e-6):.1f} "
                f"loss={float(loss.item()):.4f}",
                flush=True,
            )

    env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=min(8, max(2, os.cpu_count() or 2)))
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--run-name", type=str, default="a3c_rep_c")
    parser.add_argument("--no-shaping", action="store_true")
    args = parser.parse_args()

    if args.total_steps < 1 or args.workers < 1 or args.rollout_steps < 1:
        raise ValueError("total-steps, workers and rollout-steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be > 0")

    args.shaping = not args.no_shaping
    ctx = mp.get_context("spawn")

    shared_model = ActorCritic()
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.learning_rate)
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()

    processes: list[mp.Process] = []
    for rank in range(args.workers):
        process = ctx.Process(
            target=_worker,
            args=(rank, shared_model, optimizer, counter, lock, args),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"A3C worker exited with code {process.exitcode}")

    out_dir = MODELS_DIR / "a3c_rep_c_experiments" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": shared_model.state_dict(),
            "obs_dim": REPRESENTATION_C_DIM,
            "n_actions": JOINT_ACTION_SIZE,
            "hidden_sizes": [128, 128],
            "total_steps": args.total_steps,
            "workers": args.workers,
            "rollout_steps": args.rollout_steps,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "seed": args.seed,
        },
        out_dir / "a3c_model.pt",
    )
    print(f"Saved A3C model to {out_dir / 'a3c_model.pt'}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
