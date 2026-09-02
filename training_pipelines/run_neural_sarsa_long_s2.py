"""Focused long-run Neural SARSA experiment.

Extends the best portfolio-screen configuration (s2) from 200k to 500k
transitions at two independent seeds. Uses the canonical 76-D Representation B
and the existing true-SARSA training implementation. Outputs remain isolated
from protected/final submission artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.common.env_factory import make_eval_env, make_training_env
from src.algorithms.neural_sarsa import NeuralSarsaConfig, train_neural_sarsa

MODELS_DIR = TRAINING_ROOT / "models" / "neural_sarsa_long_s2"
RESULTS_DIR = TRAINING_ROOT / "eval_results" / "neural_sarsa_long_s2"

CONFIG = {
    "learning_rate": 2e-4,
    "gamma": 0.99,
    "epsilon_end": 0.02,
    "batch_size": 128,
    "buffer_size": 100_000,
    "target_update_interval": 2_000,
}

DEFAULT_TRANSITIONS = 500_000
DEFAULT_SEEDS = (20260903, 20260904)


def train_one(seed: int, total_transitions: int) -> dict:
    run_id = f"s2_500k_seed{seed}"
    save_dir = MODELS_DIR / run_id
    cfg = NeuralSarsaConfig(
        gamma=CONFIG["gamma"],
        learning_rate=CONFIG["learning_rate"],
        epsilon_end=CONFIG["epsilon_end"],
        batch_size=CONFIG["batch_size"],
        buffer_size=CONFIG["buffer_size"],
        target_update_interval=CONFIG["target_update_interval"],
        epsilon_decay_transitions=int(0.8 * total_transitions),
        seed=seed,
        total_transitions=total_transitions,
        eval_every=25_000,
        eval_seeds=tuple(range(900, 910)),
    )

    def env_fn():
        return make_training_env(
            shaping=True,
            shaping_kwargs={"anneal_steps": total_transitions},
        )

    def eval_env_fn():
        return make_eval_env()

    result = train_neural_sarsa(
        env_fn,
        eval_env_fn,
        save_dir,
        cfg,
        verbose=True,
    )
    best = min(result["history"], key=lambda row: row["eval_mean_cost"])
    row = {
        "run_id": run_id,
        "seed": seed,
        "total_transitions": total_transitions,
        **CONFIG,
        "best_eval_cost": float(result["best_eval_cost"]),
        "best_eval_service": float(best["eval_mean_service"]),
        "best_eval_step": int(best["step"]),
        "final_eval_cost": float(result["history"][-1]["eval_mean_cost"]),
        "run_dir": str(save_dir),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-transitions", type=int, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    print(
        f"Running focused Neural SARSA s2 for {args.total_transitions:,} transitions "
        f"with seeds={args.seeds}",
        flush=True,
    )
    print(f"Configuration: {CONFIG}", flush=True)

    for seed in args.seeds:
        print(f"\n=== {seed} ===", flush=True)
        row = train_one(int(seed), args.total_transitions)
        results.append(row)
        print(
            f"BEST internal cost={row['best_eval_cost']:,.2f} "
            f"service={row['best_eval_service']:.6f} "
            f"step={row['best_eval_step']:,}",
            flush=True,
        )

    output = RESULTS_DIR / "long_run_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Neural SARSA long-run s2 ranking ===")
    for rank, row in enumerate(sorted(results, key=lambda r: r["best_eval_cost"]), 1):
        print(
            f"{rank:2d}. {row['run_id']} "
            f"cost={row['best_eval_cost']:,.2f} "
            f"service={row['best_eval_service']:.6f} "
            f"step={row['best_eval_step']:,}"
        )
    print(f"\nSaved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
