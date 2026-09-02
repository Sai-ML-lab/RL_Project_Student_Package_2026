"""Evaluate a saved DQN + Representation C model on the official holdout suite."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from training_pipelines.src.features.representation_c import flatten_observation_representation_c
from training_pipelines.training_utils.action_wrapper import joint_index_to_quantities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = DQN.load(str(args.model), device="cpu")

    def policy(observation):
        features = flatten_observation_representation_c(observation)
        action, _ = model.predict(features, deterministic=True)
        return joint_index_to_quantities(int(action.item() if hasattr(action, "item") else action))

    seeds = HOLDOUT_SEEDS[: max(1, min(args.seeds, len(HOLDOUT_SEEDS)))]
    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=seeds,
        scenario_modes=SCENARIO_MODES,
        domain_randomization=True,
        progress=True,
    )
    overall = summarise_overall(per_episode)

    print("\n=== DQN Rep-C holdout ===")
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
