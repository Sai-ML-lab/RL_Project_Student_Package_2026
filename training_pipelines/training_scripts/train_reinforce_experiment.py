"""REINFORCE with learned baseline for the IITM RL inventory project.

Technique
---------
REINFORCE with a learned state-value baseline.

This is intentionally distinct from A2C:
    * complete episodes are collected before updating;
    * returns are Monte-Carlo returns G_t;
    * the baseline is fitted to those complete returns;
    * there is no bootstrapping and no GAE;
    * the policy gradient uses (G_t - V(s_t)).

The experiment is isolated from all official submission artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training_scripts.common import (
    build_training_vec_env,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = (
    PROJECT_ROOT / "models" / "reinforce_experiments"
)

RESULTS_DIR = (
    PROJECT_ROOT / "eval_results" / "reinforce_experiments"
)


# ---------------------------------------------------------------------------
# Fixed experiment defaults
# ---------------------------------------------------------------------------

SEED = 20260727

OBS_DIM = 35
N_PRODUCTS = 3
N_ACTIONS_PER_PRODUCT = 11

HIDDEN_SIZES = (128, 128)

LEARNING_RATE = 3e-4
VALUE_LEARNING_RATE = 3e-4

GAMMA = 0.99

ENTROPY_COEF = 0.01

GRADIENT_CLIP = 1.0

EPISODE_LENGTH = 50


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class ReinforcePolicy(nn.Module):
    """Shared MLP with three categorical action heads."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        last_dim = obs_dim

        for hidden_dim in hidden_sizes:
            layers.append(
                nn.Linear(
                    last_dim,
                    hidden_dim,
                )
            )
            layers.append(
                nn.ReLU()
            )
            last_dim = hidden_dim

        self.trunk = nn.Sequential(
            *layers
        )

        self.action_heads = nn.ModuleList(
            [
                nn.Linear(
                    last_dim,
                    N_ACTIONS_PER_PRODUCT,
                )
                for _ in range(N_PRODUCTS)
            ]
        )

    def logits(
        self,
        observations: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden = self.trunk(
            observations
        )

        return [
            head(hidden)
            for head in self.action_heads
        ]

    def sample_action(
        self,
        observation: torch.Tensor,
    ):
        logits = self.logits(
            observation
        )

        distributions = [
            torch.distributions.Categorical(
                logits=head_logits
            )
            for head_logits in logits
        ]

        actions = torch.stack(
            [
                distribution.sample()
                for distribution in distributions
            ],
            dim=-1,
        )

        log_prob = sum(
            distribution.log_prob(
                actions[:, product_index]
            )
            for product_index, distribution
            in enumerate(distributions)
        )

        entropy = sum(
            distribution.entropy()
            for distribution in distributions
        )

        return (
            actions,
            log_prob,
            entropy,
        )

    def greedy_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.logits(
            observation
        )

        return torch.stack(
            [
                torch.argmax(
                    head_logits,
                    dim=-1,
                )
                for head_logits in logits
            ],
            dim=-1,
        )


class ValueNetwork(nn.Module):
    """State-value baseline V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        last_dim = obs_dim

        for hidden_dim in hidden_sizes:
            layers.append(
                nn.Linear(
                    last_dim,
                    hidden_dim,
                )
            )
            layers.append(
                nn.ReLU()
            )
            last_dim = hidden_dim

        self.network = nn.Sequential(
            *layers,
            nn.Linear(
                last_dim,
                1,
            ),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            observations
        ).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def observation_to_tensor(
    observation,
) -> torch.Tensor:
    """Convert an already-flattened environment observation to a tensor.

    build_training_vec_env() applies FlattenObs, so the observation received
    here is already a numeric 35-dimensional vector.
    """
    features = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    if features.shape != (OBS_DIM,):
        raise ValueError(
            "Unexpected flattened observation shape: "
            f"{features.shape}; expected {(OBS_DIM,)}"
        )

    return torch.from_numpy(
        features
    ).unsqueeze(0)


def compute_returns(
    rewards: list[float],
    gamma: float,
) -> np.ndarray:
    """Compute Monte-Carlo discounted returns."""

    returns = np.zeros(
        len(rewards),
        dtype=np.float32,
    )

    running_return = 0.0

    for index in range(
        len(rewards) - 1,
        -1,
        -1,
    ):
        running_return = (
            float(rewards[index])
            + gamma * running_return
        )

        returns[index] = (
            running_return
        )

    return returns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_run(
    *,
    total_transitions: int,
    seed: int,
    save_dir: Path,
) -> dict:

    if total_transitions < EPISODE_LENGTH:
        raise ValueError(
            "total_transitions must be at least "
            f"{EPISODE_LENGTH}."
        )

    if total_transitions % EPISODE_LENGTH != 0:
        raise ValueError(
            "For this experiment, total_transitions must "
            "be divisible by the episode length of 50."
        )

    seed_everything(
        seed
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    actor = ReinforcePolicy(
        OBS_DIM
    )

    baseline = ValueNetwork(
        OBS_DIM
    )

    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=LEARNING_RATE,
    )

    baseline_optimizer = optim.Adam(
        baseline.parameters(),
        lr=VALUE_LEARNING_RATE,
    )

    env = build_training_vec_env(
        n_envs=1,
        flatten_action=False,
        shaping=True,
        seed=seed,
        shaping_kwargs={
            "anneal_steps": total_transitions,
        },
    )

    episodes_target = (
        total_transitions
        // EPISODE_LENGTH
    )

    transition_count = 0

    episode_returns: list[float] = []

    print(
        "\n=== REINFORCE WITH BASELINE ===",
        flush=True,
    )

    print(
        f"transitions={total_transitions:,} "
        f"episodes={episodes_target:,} "
        f"gamma={GAMMA} "
        f"lr={LEARNING_RATE} "
        f"baseline_lr={VALUE_LEARNING_RATE} "
        f"entropy_coef={ENTROPY_COEF}",
        flush=True,
    )

    try:

        for episode_index in range(
            episodes_target
        ):

            obs = env.reset()

            observations: list[
                np.ndarray
            ] = []

            actions: list[
                np.ndarray
            ] = []

            rewards: list[
                float
            ] = []

            done = False

            while not done:

                observation_tensor = (
                    observation_to_tensor(
                        obs[0]
                    )
                )

                with torch.no_grad():
                    sampled_action, _, _ = (
                        actor.sample_action(
                            observation_tensor
                        )
                    )

                action_np = (
                    sampled_action
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                if action_np.shape != (
                    N_PRODUCTS,
                ):
                    raise ValueError(
                        "Unexpected action shape: "
                        f"{action_np.shape}"
                    )

                if np.any(
                    action_np < 0
                ) or np.any(
                    action_np > 10
                ):
                    raise ValueError(
                        "Sampled invalid action indices: "
                        f"{action_np.tolist()}"
                    )

                observations.append(
                    observation_tensor.squeeze(0)
                    .numpy()
                )

                actions.append(
                    action_np.copy()
                )

                obs, reward, dones, infos = (
                    env.step(
                        action_np.reshape(
                            1,
                            N_PRODUCTS,
                        )
                    )
                )

                rewards.append(
                    float(reward[0])
                )

                transition_count += 1

                done = bool(
                    dones[0]
                )

                if transition_count >= total_transitions:
                    done = True

            # ---------------------------------------------------------------
            # Monte-Carlo return
            # ---------------------------------------------------------------

            returns_np = compute_returns(
                rewards,
                GAMMA,
            )

            observations_tensor = torch.tensor(
                np.asarray(
                    observations,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

            actions_tensor = torch.tensor(
                np.asarray(
                    actions,
                    dtype=np.int64,
                ),
                dtype=torch.long,
            )

            returns_tensor = torch.tensor(
                returns_np,
                dtype=torch.float32,
            )

            # ---------------------------------------------------------------
            # Baseline update
            #
            # Critic is fitted to COMPLETE Monte-Carlo returns.
            # There is no bootstrap target.
            # ---------------------------------------------------------------

            baseline_optimizer.zero_grad()

            values = baseline(
                observations_tensor
            )

            baseline_loss = nn.functional.mse_loss(
                values,
                returns_tensor,
            )

            baseline_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                baseline.parameters(),
                GRADIENT_CLIP,
            )

            baseline_optimizer.step()

            # ---------------------------------------------------------------
            # REINFORCE policy update
            #
            # advantage = G_t - V(s_t)
            # ---------------------------------------------------------------

            with torch.no_grad():
                advantages = (
                    returns_tensor
                    - baseline(
                        observations_tensor
                    )
                )

                # Normalize advantages for numerical stability.
                advantages = (
                    advantages
                    - advantages.mean()
                ) / (
                    advantages.std()
                    + 1e-8
                )

            actor_optimizer.zero_grad()

            logits = actor.logits(
                observations_tensor
            )

            log_prob = torch.zeros(
                observations_tensor.shape[0],
                dtype=torch.float32,
            )

            entropy = torch.zeros_like(
                log_prob
            )

            for product_index in range(
                N_PRODUCTS
            ):

                distribution = (
                    torch.distributions.Categorical(
                        logits=logits[
                            product_index
                        ]
                    )
                )

                product_actions = (
                    actions_tensor[
                        :,
                        product_index
                    ]
                )

                log_prob = (
                    log_prob
                    + distribution.log_prob(
                        product_actions
                    )
                )

                entropy = (
                    entropy
                    + distribution.entropy()
                )

            policy_loss = -(
                log_prob
                * advantages
            ).mean()

            entropy_bonus = (
                entropy.mean()
            )

            total_actor_loss = (
                policy_loss
                - ENTROPY_COEF
                * entropy_bonus
            )

            total_actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                GRADIENT_CLIP,
            )

            actor_optimizer.step()

            episode_return = float(
                np.sum(rewards)
            )

            episode_returns.append(
                episode_return
            )

            if (
                (episode_index + 1) % 250 == 0
                or episode_index == 0
                or episode_index + 1
                == episodes_target
            ):
                recent_returns = (
                    episode_returns[-250:]
                )

                print(
                    f"[episode "
                    f"{episode_index + 1:>5}/"
                    f"{episodes_target}] "
                    f"transition={transition_count:>7,} "
                    f"episode_return="
                    f"{episode_return:,.2f} "
                    f"mean250="
                    f"{np.mean(recent_returns):,.2f} "
                    f"policy_loss="
                    f"{policy_loss.item():.4f} "
                    f"baseline_loss="
                    f"{baseline_loss.item():.4f}",
                    flush=True,
                )

    finally:
        env.close()

    # -----------------------------------------------------------------------
    # Save actor only for inference.
    # -----------------------------------------------------------------------

    actor_path = (
        save_dir / "actor_state.pt"
    )

    torch.save(
        {
            "state_dict": actor.state_dict(),
            "obs_dim": OBS_DIM,
            "hidden_sizes": HIDDEN_SIZES,
            "n_products": N_PRODUCTS,
            "n_actions_per_product": N_ACTIONS_PER_PRODUCT,
            "technique": (
                "REINFORCE with learned baseline"
            ),
        },
        actor_path,
    )

    # Save baseline separately for experiment reproducibility.
    baseline_path = (
        save_dir / "baseline_state.pt"
    )

    torch.save(
        {
            "state_dict": baseline.state_dict(),
            "obs_dim": OBS_DIM,
            "hidden_sizes": HIDDEN_SIZES,
        },
        baseline_path,
    )

    result = {
        "technique": (
            "REINFORCE with learned baseline"
        ),
        "seed": seed,
        "total_transitions": total_transitions,
        "episodes": episodes_target,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "value_learning_rate": VALUE_LEARNING_RATE,
        "entropy_coef": ENTROPY_COEF,
        "hidden_sizes": list(
            HIDDEN_SIZES
        ),
        "final_episode_return": (
            episode_returns[-1]
            if episode_returns
            else None
        ),
        "mean_last_250_episode_returns": (
            float(
                np.mean(
                    episode_returns[-250:]
                )
            )
            if episode_returns
            else None
        ),
        "actor_path": str(
            actor_path
        ),
        "baseline_path": str(
            baseline_path
        ),
    }

    result_path = (
        RESULTS_DIR
        / (
            f"reinforce_{total_transitions // 1000}k"
            f"_seed{seed}.json"
        )
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== REINFORCE experiment complete ===",
        flush=True,
    )

    print(
        f"Actor checkpoint : {actor_path}",
        flush=True,
    )

    print(
        f"Baseline checkpoint : {baseline_path}",
        flush=True,
    )

    print(
        f"Result file : {result_path}",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Train REINFORCE with a learned baseline."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.timesteps < EPISODE_LENGTH:
        raise ValueError(
            "--timesteps must be >= 50."
        )

    run_name = (
        args.run_name
        if args.run_name
        else (
            f"reinforce_{args.timesteps // 1000}k"
        )
    )

    save_dir = (
        EXPERIMENTS_DIR
        / f"{run_name}_seed{args.seed}"
    )

    train_one_run(
        total_transitions=args.timesteps,
        seed=args.seed,
        save_dir=save_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )