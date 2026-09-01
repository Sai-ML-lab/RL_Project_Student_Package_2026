"""Promote the strongest PPO screening configurations.

The screening file is validated before promotion. Configurations are ranked
by official mean episode cost, where lower cost is better.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_ppo_screening import (
    MODELS_DIR,
    RESULTS_DIR,
    train_one_config,
)


PROMOTE_TIMESTEPS = 1_500_000


def _load_screening_results(
    path: Path,
) -> list[dict]:
    """Load and validate PPO screening results."""

    if not path.exists():
        raise FileNotFoundError(
            f"PPO screening results not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        results = json.load(handle)

    if not isinstance(results, list):
        raise ValueError(
            "PPO screening results must be a list."
        )

    if not results:
        raise ValueError(
            "PPO screening results are empty."
        )

    required_fields = {
        "config_id",
        "ent_coef",
        "learning_rate",
        "mean_cost",
        "mean_service_level",
    }

    for row in results:
        missing = required_fields.difference(
            row.keys()
        )

        if missing:
            raise ValueError(
                "Screening result is missing fields: "
                f"{sorted(missing)}"
            )

        mean_cost = float(
            row["mean_cost"]
        )

        mean_service = float(
            row["mean_service_level"]
        )

        if not math.isfinite(mean_cost):
            raise ValueError(
                "Non-finite mean_cost found in PPO "
                f"screening results: {row}"
            )

        if not math.isfinite(mean_service):
            raise ValueError(
                "Non-finite mean_service_level found in PPO "
                f"screening results: {row}"
            )

    return sorted(
        results,
        key=lambda row: float(
            row["mean_cost"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote the best PPO screening configurations."
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=PROMOTE_TIMESTEPS,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError(
            "--timesteps must be >= 1."
        )

    if args.top_n < 1:
        raise ValueError(
            "--top-n must be >= 1."
        )

    screening_path = (
        RESULTS_DIR / "ppo_screening_results.json"
    )

    screening_results = _load_screening_results(
        screening_path
    )

    top_configs = screening_results[
        : args.top_n
    ]

    print(
        f"Promoting top {len(top_configs)} "
        f"configurations to {args.timesteps:,} timesteps:"
    )

    for row in top_configs:
        print(
            f"  {row['config_id']}: "
            f"ent_coef={row['ent_coef']} "
            f"lr={row['learning_rate']} "
            f"screening_cost={row['mean_cost']:,.1f}"
        )

    final_results: list[dict] = []

    for row in top_configs:
        ent_coef = float(
            row["ent_coef"]
        )

        learning_rate = float(
            row["learning_rate"]
        )

        config_id = (
            f"promoted_ent{ent_coef}"
            f"_lr{learning_rate}"
        )

        save_dir = (
            MODELS_DIR / config_id
        )

        print(
            f"\n=== training {config_id} "
            f"({args.timesteps:,} timesteps) ===",
            flush=True,
        )

        result = train_one_config(
            ent_coef=ent_coef,
            lr=learning_rate,
            timesteps=args.timesteps,
            save_dir=save_dir,
        )

        result["config_id"] = config_id
        result["source_screening_config"] = (
            row["config_id"]
        )

        final_results.append(result)

        print(
            f"  mean_cost={result['mean_cost']:,.1f} "
            f"service={result['mean_service_level']:.4f}",
            flush=True,
        )

    final_results.sort(
        key=lambda row: row["mean_cost"]
    )

    output_path = (
        RESULTS_DIR / "ppo_promoted_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_results,
            handle,
            indent=2,
        )

    print(
        "\n=== Final promoted results "
        "(lowest official cost first) ==="
    )

    for rank, row in enumerate(
        final_results,
        start=1,
    ):
        print(
            f"{rank}. {row['config_id']} "
            f"cost={row['mean_cost']:,.1f} "
            f"service={row['mean_service_level']:.4f}"
        )

    if final_results:
        print(
            f"\nChampion config: "
            f"{final_results[0]['config_id']}"
        )


if __name__ == "__main__":
    main()