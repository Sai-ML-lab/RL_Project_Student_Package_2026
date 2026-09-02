"""Evaluate a saved A2C model on the official holdout suite."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from training_pipelines.training_utils.obs_wrapper import flatten_observation


EXPECTED_FEATURE_DIM = 35


def _resolve_model_path(path: Path) -> Path:
    """Accept either a checkpoint file or a run directory."""
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "best_model.zip", path / "final_model.zip"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No A2C checkpoint found inside {path}. "
            "Expected best_model.zip or final_model.zip."
        )
    raise FileNotFoundError(f"Model path not found: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    model = A2C.load(str(model_path), device="cpu")

    # A2C in this pipeline is trained with the canonical 35-D FlattenObs
    # representation. Do not pass the 76-D Representation B features here.
    def policy(observation):
        features = flatten_observation(observation)

        if features.shape != (EXPECTED_FEATURE_DIM,):
            raise ValueError(
                f"Unexpected A2C feature shape: {features.shape}; "
                f"expected ({EXPECTED_FEATURE_DIM},)"
            )
        if not np.isfinite(features).all():
            raise ValueError("A2C features contain non-finite values")

        action, _ = model.predict(features, deterministic=True)
        values = [int(x) for x in np.asarray(action).reshape(-1)]

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
