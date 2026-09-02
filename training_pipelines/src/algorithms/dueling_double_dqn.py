"""Double DQN with a dueling joint-action network for the IITM inventory-control environment.

The environment uses the full joint Discrete(1331) action space:

    joint_index = a1 * 121 + a2 * 11 + a3

where each ai is in {0, ..., 10}.

The network uses a dueling decomposition:

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

The target uses the genuine Double-DQN formulation:

    a* = argmax_a Q_online(s', a)
    y  = r + gamma * Q_target(s', a*)

This preserves interactions between products and does not assume that the
scalar environment reward is additively separable across products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling MLP over all 1331 joint actions."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        if len(hidden_sizes) != 2:
            raise ValueError(
                "DuelingQNetwork expects exactly two hidden sizes."
            )

        h1, h2 = hidden_sizes

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(h2, 1)
        self.advantage_head = nn.Linear(h2, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(obs)

        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)

        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values


@dataclass
class ReplayBuffer:
    """Simple fixed-size replay buffer for vector observations."""

    capacity: int
    obs_dim: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")

        if self.obs_dim <= 0:
            raise ValueError("Observation dimension must be positive.")

        self.obs = np.zeros(
            (self.capacity, self.obs_dim),
            dtype=np.float32,
        )

        self.next_obs = np.zeros(
            (self.capacity, self.obs_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            self.capacity,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.dones = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected obs shape {(self.obs_dim,)}, got {obs.shape}."
            )

        if next_obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected next_obs shape {(self.obs_dim,)}, "
                f"got {next_obs.shape}."
            )

        idx = self.pos

        self.obs[idx] = obs
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if not 1 <= batch_size <= self.size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} "
                f"from replay size={self.size}."
            )

        indices = rng.integers(
            0,
            self.size,
            size=batch_size,
        )

        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }


def linear_epsilon(
    step: int,
    total_steps: int,
    initial: float = 1.0,
    final: float = 0.02,
    fraction: float = 0.30,
) -> float:
    """Linearly decay epsilon over a fraction of total training."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")

    decay_steps = max(
        1,
        int(total_steps * fraction),
    )

    clipped_step = min(
        max(step, 0),
        decay_steps,
    )

    progress = clipped_step / decay_steps

    return float(
        initial
        + (final - initial) * progress
    )


def select_action(
    q_net: DuelingQNetwork,
    obs: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    n_actions: int,
    device: torch.device,
) -> int:
    """Epsilon-greedy joint-action selection."""

    if rng.random() < epsilon:
        return int(
            rng.integers(
                0,
                n_actions,
            )
        )

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = q_net(obs_tensor)
        return int(
            q_values.argmax(dim=1).item()
        )


def double_dqn_update(
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Perform one Double-DQN update."""

    obs_t = torch.as_tensor(
        batch["obs"],
        dtype=torch.float32,
        device=device,
    )

    actions_t = torch.as_tensor(
        batch["actions"],
        dtype=torch.long,
        device=device,
    )

    rewards_t = torch.as_tensor(
        batch["rewards"],
        dtype=torch.float32,
        device=device,
    )

    next_obs_t = torch.as_tensor(
        batch["next_obs"],
        dtype=torch.float32,
        device=device,
    )

    dones_t = torch.as_tensor(
        batch["dones"],
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------------
    # Double DQN target:
    #
    # 1. Online network chooses the next action.
    # 2. Target network evaluates that chosen action.
    # --------------------------------------------------------------
    with torch.no_grad():
        online_next_q = online(next_obs_t)

        next_actions = online_next_q.argmax(
            dim=1,
            keepdim=True,
        )

        target_next_q = target(next_obs_t).gather(
            dim=1,
            index=next_actions,
        ).squeeze(1)

        td_target = (
            rewards_t
            + gamma
            * (1.0 - dones_t)
            * target_next_q
        )

    current_q = online(obs_t).gather(
        dim=1,
        index=actions_t.unsqueeze(1),
    ).squeeze(1)

    loss = F.smooth_l1_loss(
        current_q,
        td_target,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        online.parameters(),
        max_grad_norm,
    )

    optimizer.step()

    return float(loss.item())