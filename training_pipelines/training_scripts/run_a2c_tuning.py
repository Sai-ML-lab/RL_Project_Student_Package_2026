"""Focused A2C hyperparameter screening on the canonical 76-D Representation B."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE = "training_pipelines.training_scripts.train_a2c"
EVAL_MODULE = "training_pipelines.training_scripts.evaluate_a2c"
RESULTS_DIR = PROJECT_ROOT / "tuning_results" / "a2c"
MODEL_ROOT = PROJECT_ROOT / "training_pipelines" / "models" / "a2c_experiments"

CONFIGS = [
    {"learning_rate": 3e-4, "gamma": 0.98, "gae_lambda": 1.0, "ent_coef": 0.001, "n_steps": 16},
    {"learning_rate": 5e-4, "gamma": 0.98, "gae_lambda": 1.0, "ent_coef": 0.005, "n_steps": 32},
    {"learning_rate": 7e-4, "gamma": 0.98, "gae_lambda": 1.0, "ent_coef": 0.01, "n_steps": 32},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "n_steps": 32},
    {"learning_rate": 7e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "n_steps": 32},
    {"learning_rate": 5e-4, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.001, "n_steps": 64},
]


def _evaluate(model_path: Path, seed_count: int) -> dict[str, float]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            EVAL_MODULE,
            "--model",
            str(model_path),
            "--seeds",
            str(seed_count),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    cost_match = re.search(r"mean_cost=([0-9.,]+)", completed.stdout)
    service_match = re.search(r"mean_service_level=([0-9.]+)", completed.stdout)
    if cost_match is None:
        raise RuntimeError("Could not parse A2C mean_cost from evaluator output")
    return {
        "quick_mean_cost": float(cost_match.group(1).replace(",", "")),
        "quick_mean_service_level": float(service_match.group(1)) if service_match else float("nan"),
    }


def _run(config: dict, steps: int, seed: int, n_envs: int, eval_seeds: int) -> dict:
    tag = (
        f"lr{config['learning_rate']:.0e}_g{config['gamma']}_"
        f"lam{config['gae_lambda']}_ent{config['ent_coef']}_n{config['n_steps']}"
    )
    run_name = f"screen_{tag}_seed{seed}"
    cmd = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--timesteps", str(steps),
        "--n-envs", str(n_envs),
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"\n=== A2C {run_name} ===", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    model_path = MODEL_ROOT / run_name / "final_model.zip"
    if not model_path.exists():
        model_path = MODEL_ROOT / run_name / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No A2C checkpoint found for {run_name}")

    scores = _evaluate(model_path, eval_seeds)
    row = {
        "run_name": run_name,
        "model": str(model_path),
        "steps": steps,
        "n_envs": n_envs,
        "eval_seeds": eval_seeds,
        "config": config,
        **scores,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"quick_mean_cost={scores['quick_mean_cost']:,.2f} "
        f"quick_service={scores['quick_mean_service_level']:.6f}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=10)
    args = parser.parse_args()

    if min(args.steps, args.n_envs, args.max_runs, args.eval_seeds) < 1:
        raise ValueError("steps, n-envs, max-runs and eval-seeds must be positive")

    rows = []
    for config in CONFIGS[:args.max_runs]:
        rows.append(_run(config, args.steps, args.seed, args.n_envs, args.eval_seeds))

    rows.sort(key=lambda row: row["quick_mean_cost"])
    manifest = RESULTS_DIR / "screening_manifest.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== A2C SCREENING RANKING ===")
    for rank, row in enumerate(rows, 1):
        c = row["config"]
        print(
            f"{rank:>2}. cost={row['quick_mean_cost']:>10,.2f} "
            f"service={row['quick_mean_service_level']:.6f} "
            f"lr={c['learning_rate']} gamma={c['gamma']} "
            f"lambda={c['gae_lambda']} ent={c['ent_coef']} n_steps={c['n_steps']}"
        )
    print(f"Saved A2C screening manifest to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
