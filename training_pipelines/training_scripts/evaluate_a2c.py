"""Evaluate a saved A2C model on the official holdout suite."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from training_pipelines.src.features.engineered import flatten_observation_representation_b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = A2C.load(str(args.model), device="cpu")

    def policy(observation):
        features = flatten_observation_representation_b(observation)
        action, _ = model.predict(features, deterministic=True)
        values = [int(x) for x in action.tolist()]
        if len(values) != 3 or any(x < 0 or x > 10 for x in values):
            raise ValueError(f"Unexpected A2C action: {values}")
        return [x * 10 for x in values]

    seeds = HOLDOUT_SEEDS[: max(1, min(args.seeds, len(HOLDOUT_SEEDS)))]
    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=seeds,
        scenario_modes=SCENARIO_MODES,
        domain_randomization=True,
        progress=True,
    )
    overall = summarise_overall(per_episode)

    print("\n=== A2C holdout ===")
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
