"""Small A3C hyperparameter screen for Representation C.

Runs a compact set of configurations rather than a large grid. Each model is
evaluated on a 50-episode quick holdout (10 seeds x 5 scenarios) before the
best configuration is promoted to the full 200-episode evaluation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from evaluation import SCENARIO_MODES, evaluate_policy, summarise_overall

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "training_pipelines" / "training_scripts" / "train_a3c_rep_c.py"
EVAL_SCRIPT = PROJECT_ROOT / "training_pipelines" / "training_scripts" / "evaluate_a3c_rep_c.py"
RESULTS_DIR = PROJECT_ROOT / "tuning_results" / "a3c"

CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.98, "gae_lambda": 0.95, "entropy_coef": 0.001, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.98, "gae_lambda": 0.95, "entropy_coef": 0.005, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.005, "rollout_steps": 20},
    {"learning_rate": 5e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.001, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 1.00, "entropy_coef": 0.005, "rollout_steps": 10},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.90, "entropy_coef": 0.001, "rollout_steps": 40},
]


def _run(config: dict, steps: int, seed: int, workers: int) -> dict:
    tag = (
        f"lr{config['learning_rate']:.0e}_g{config['gamma']}_"
        f"lam{config['gae_lambda']}_ent{config['entropy_coef']}_r{config['rollout_steps']}"
    )
    run_name = f"screen_{tag}_seed{seed}"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--total-steps", str(steps),
        "--workers", str(workers),
        "--seed", str(seed),
        "--run-name", run_name,
        "--normalize-advantage",
    ]
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"\n=== A3C {run_name} ===", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    model_path = PROJECT_ROOT / "models" / "a3c_rep_c_experiments" / run_name / "a3c_model.pt"
    # Reuse the official evaluator in a short form by invoking it from Python.
    payload = {
        "model": str(model_path),
        "config": config,
        "run_name": run_name,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-runs", type=int, default=6)
    args = parser.parse_args()

    # Training is the expensive part. The script leaves official holdout
    # selection to evaluate_a3c_rep_c.py so users can choose 10 seeds for
    # screening or all 40 seeds for final promotion.
    selected = CONFIGS[: max(1, args.max_runs)]
    rows = []
    for cfg in selected:
        result = _run(cfg, args.steps, args.seed, args.workers)
        result["steps"] = args.steps
        result["workers"] = args.workers
        rows.append(result)

    out = RESULTS_DIR / "screening_manifest.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved A3C screening manifest to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
