"""Evaluate a saved A3C + Representation C model on the official holdout suite."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from training_pipelines.src.environment.action_codec import JOINT_ACTION_SIZE, joint_index_to_quantities
from training_pipelines.src.features.representation_c import (
    REPRESENTATION_C_DIM,
    flatten_observation_representation_c,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    payload = torch.load(args.model, map_location="cpu")
    model = ActorCritic(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload["n_actions"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def policy(observation):
        features = flatten_observation_representation_c(observation)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(features).float().unsqueeze(0))
        action_index = int(torch.argmax(logits, dim=-1).item())
        return joint_index_to_quantities(action_index)

    seeds = HOLDOUT_SEEDS[: max(1, min(args.seeds, len(HOLDOUT_SEEDS)))]
    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=seeds,
        scenario_modes=SCENARIO_MODES,
        domain_randomization=True,
        progress=True,
    )
    overall = summarise_overall(per_episode)

    print("\n=== A3C Rep-C holdout ===")
    print(f"episodes={overall['n_episodes']}")
    print(f"mean_cost={overall['mean_cost']:,.2f}")
    print(f"std_cost={overall['std_cost']:,.2f}")
    print(f"median_cost={overall['median_cost']:,.2f}")
    print(f"mean_service_level={overall['mean_service_level']:.6f}")
    print("\nScenario summary:")
    print(scenario_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
