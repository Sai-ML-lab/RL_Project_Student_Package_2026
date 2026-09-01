
"""Neural Network SARSA (Phase 4, section 7).

The defining difference from DQN: the TD target evaluates Q at the action
the behavior (epsilon-greedy) policy ACTUALLY takes next, never the greedy
argmax action. A replay buffer is used for sample efficiency; each stored
transition keeps the true next action taken at collection time so the SARSA
target is preserved even though updates are sampled out of order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.action_codec import JOINT_ACTION_SIZE

from .common.checkpoint import save_checkpoint
from .common.networks import DEFAULT_HIDDEN_SIZES, QNetwork
from .common.replay_buffer import SarsaReplayBuffer
from .common.schedules import linear_schedule


@dataclass
class NeuralSarsaConfig:
    gamma: float = 0.99
    learning_rate: float = 2e-4
    batch_size: int = 128
    buffer_size: int = 50_000
    learning_starts: int = 5_000
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_transitions: int = 300_000
    max_grad_norm: float = 10.0
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    n_actions: int = JOINT_ACTION_SIZE
    seed: int = 20260825
    total_transitions: int = 250_000
    eval_every: int = 25_000
    eval_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(900, 910)))


def _epsilon_greedy_action(
    q_values: np.ndarray, epsilon: float, rng: np.random.Generator, n_actions: int
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(q_values))


def _evaluate_greedy(q_net: QNetwork, eval_env_fn: Callable[[], Any], seeds) -> tuple[float, float]:
    env = eval_env_fn()
    costs, services = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        ep_cost = 0.0
        ep_demand = ep_fulfilled = 0
        while not done:
            with torch.no_grad():
                q_values = q_net(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=-1).item())
            obs, _reward, terminated, truncated, info = env.step(action)
            ep_cost += float(info["costs"]["daily_total"])
            ep_demand += int(np.sum(info["demand"]))
            ep_fulfilled += int(np.sum(info["fulfilled_demand"]))
            done = bool(terminated or truncated)
        costs.append(ep_cost)
        services.append(ep_fulfilled / max(ep_demand, 1))
    return float(np.mean(costs)), float(np.mean(services))


def train_neural_sarsa(
    env_fn: Callable[[], Any],
    eval_env_fn: Callable[[], Any],
    save_dir: str | Path,
    cfg: NeuralSarsaConfig = NeuralSarsaConfig(),
    log_writer: Callable[[dict], None] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = env_fn()
    obs_dim = int(np.asarray(env.observation_space.shape).prod())

    q_net = QNetwork(obs_dim, cfg.n_actions, cfg.hidden_sizes)
    q_target = QNetwork(obs_dim, cfg.n_actions, cfg.hidden_sizes)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)

    buffer = SarsaReplayBuffer(cfg.buffer_size, obs_dim)
    epsilon_schedule = linear_schedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_transitions)

    def _select_action(obs: np.ndarray, step: int) -> int:
        epsilon = epsilon_schedule(step)
        with torch.no_grad():
            q_values = q_net(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        return _epsilon_greedy_action(q_values, epsilon, rng, cfg.n_actions)

    history: list[dict[str, Any]] = []
    best_eval_cost = float("inf")
    n_gradient_updates = 0
    last_loss = float("nan")

    obs, _ = env.reset(seed=cfg.seed)
    obs = np.asarray(obs, dtype=np.float32)
    action = _select_action(obs, 0)

    heartbeat_every = max(1, cfg.total_transitions // 50)
    started_at = time.perf_counter()

    for step in range(1, cfg.total_transitions + 1):
        next_obs, reward, terminated, truncated, _info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated or truncated)
        # The actual action the behavior policy takes next -- this is what
        # makes the update SARSA rather than Q-learning/DQN.
        next_action = _select_action(next_obs, step)

        buffer.add(obs, action, reward, next_obs, next_action, done)

        if done:
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            action = _select_action(obs, step)
        else:
            obs = next_obs
            action = next_action

        if (
            step >= cfg.learning_starts
            and step % cfg.train_frequency == 0
            and len(buffer) >= cfg.batch_size
        ):
            for _ in range(cfg.gradient_steps):
                batch = buffer.sample(cfg.batch_size, rng)
                obs_t = torch.from_numpy(batch["obs"])
                actions_t = torch.from_numpy(batch["actions"]).long()
                rewards_t = torch.from_numpy(batch["rewards"])
                next_obs_t = torch.from_numpy(batch["next_obs"])
                next_actions_t = torch.from_numpy(batch["next_actions"]).long()
                dones_t = torch.from_numpy(batch["dones"])

                with torch.no_grad():
                    next_q_all = q_target(next_obs_t)
                    next_q = next_q_all.gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
                    target = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q

                current_q_all = q_net(obs_t)
                current_q = current_q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                loss = F.smooth_l1_loss(current_q, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.max_grad_norm)
                optimizer.step()
                last_loss = float(loss.item())
                n_gradient_updates += 1

                if n_gradient_updates % cfg.target_update_interval == 0:
                    q_target.load_state_dict(q_net.state_dict())

        if verbose and step % heartbeat_every == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"  ...heartbeat step {step:>8}/{cfg.total_transitions} "
                f"({100 * step / cfg.total_transitions:5.1f}%) "
                f"elapsed={elapsed:6.1f}s fps={step / elapsed:6.1f}",
                flush=True,
            )

        if step % cfg.eval_every == 0 or step == cfg.total_transitions:
            eval_mean_cost, eval_mean_service = _evaluate_greedy(q_net, eval_env_fn, cfg.eval_seeds)
            record = {
                "step": step,
                "loss": last_loss,
                "epsilon": epsilon_schedule(step),
                "eval_mean_cost": eval_mean_cost,
                "eval_mean_service": eval_mean_service,
            }
            if eval_mean_cost < best_eval_cost:
                best_eval_cost = eval_mean_cost
                save_checkpoint(
                    save_dir / "policy_state.pt",
                    model_state=q_net.state_dict(),
                    obs_dim=obs_dim,
                    n_actions=cfg.n_actions,
                    hidden_sizes=cfg.hidden_sizes,
                    extra={"step": step, "eval_mean_cost": eval_mean_cost},
                )
                record["saved_best"] = True
            history.append(record)
            if log_writer is not None:
                log_writer(record)
            if verbose:
                print(
                    f"[step {step:>8}] loss={last_loss:.4f} eps={record['epsilon']:.3f} "
                    f"eval_cost={eval_mean_cost:,.1f} eval_service={eval_mean_service:.3f}"
                    + (" *new best*" if record.get("saved_best") else ""),
                    flush=True,
                )

    save_checkpoint(
        save_dir / "policy_state_final.pt",
        model_state=q_net.state_dict(),
        obs_dim=obs_dim,
        n_actions=cfg.n_actions,
        hidden_sizes=cfg.hidden_sizes,
        extra={"step": cfg.total_transitions},
    )
    return {"history": history, "best_eval_cost": best_eval_cost}