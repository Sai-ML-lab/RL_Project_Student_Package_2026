"""Targeted Neural SARSA experiment.

Runs one explicitly specified Neural SARSA configuration and stores the
candidate in a separate experiment directory so existing submitted artifacts
are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.algorithms.common.env_factory import (
    make_eval_env,
    make_training_env,
)
from src.algorithms.neural_sarsa import (
    NeuralSarsaConfig,
    train_neural_sarsa,
)


PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "neural_sarsa_experiments"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "eval_results"
    / "neural_sarsa_experiments"
)

SEED = 20260825


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one targeted Neural SARSA experiment."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--total-transitions",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    if not 0.0 < args.gamma <= 1.0:
        raise ValueError(
            "--gamma must be in (0, 1]."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    if not 0.0 <= args.epsilon_end <= 1.0:
        raise ValueError(
            "--epsilon-end must be in [0, 1]."
        )

    if args.total_transitions < 1:
        raise ValueError(
            "--total-transitions must be >= 1."
        )

    run_id = (
        f"gamma{args.gamma}"
        f"_lr{args.learning_rate}"
        f"_eps{args.epsilon_end}"
        f"_{args.total_transitions // 1000}k"
        f"_seed{args.seed}"
    )

    save_dir = (
        MODELS_DIR
        / run_id
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = NeuralSarsaConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_end=args.epsilon_end,
        epsilon_decay_transitions=int(
            0.8 * args.total_transitions
        ),
        seed=args.seed,
        total_transitions=args.total_transitions,
    )

    def env_fn():
        return make_training_env(
            shaping=True,
            shaping_kwargs={
                "anneal_steps": args.total_transitions
            },
        )

    def eval_env_fn():
        return make_eval_env()

    print(
        "\n=== Neural SARSA targeted experiment ===",
        flush=True,
    )

    print(
        f"gamma={args.gamma} "
        f"lr={args.learning_rate} "
        f"epsilon_end={args.epsilon_end} "
        f"transitions={args.total_transitions:,} "
        f"seed={args.seed}",
        flush=True,
    )

    result = train_neural_sarsa(
        env_fn,
        eval_env_fn,
        save_dir,
        cfg,
        verbose=True,
    )

    output = {
        "run_id": run_id,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "epsilon_end": args.epsilon_end,
        "total_transitions": args.total_transitions,
        "seed": args.seed,
        "best_eval_cost": result["best_eval_cost"],
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        RESULTS_DIR
        / f"{run_id}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print(
        "\n=== Experiment complete ===",
        flush=True,
    )

    print(
        f"Best internal eval cost: "
        f"{result['best_eval_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Run directory: {save_dir}",
        flush=True,
    )

    print(
        f"Result file: {result_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())