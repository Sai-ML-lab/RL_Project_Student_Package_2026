"""Evaluate a saved DQN + Representation C model on the official holdout suite."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from training_pipelines.src.features.representation_c import flatten_observation_representation_c
from training_pipelines.training_utils.action_wrapper import joint_index_to_quantities


def _resolve_model_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("final_model.zip", "best_model.zip"):
            candidate = path / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No DQN Rep-C checkpoint found at {path}. "
        "Expected a .zip file or a directory containing final_model.zip/best_model.zip."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    model = DQN.load(str(model_path), device="cpu")

    def policy(observation):
        features = flatten_observation_representation_c(observation)
        if features.shape != (99,) or not np.isfinite(features).all():
            raise ValueError("Invalid DQN Rep-C feature vector")
        action, _ = model.predict(features, deterministic=True)
        joint_index = int(np.asarray(action).reshape(-1)[0])
        return joint_index_to_quantities(joint_index)

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
