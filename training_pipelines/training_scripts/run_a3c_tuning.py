"""A3C hyperparameter screening for Representation C.

Each selected configuration is trained for a meaningful budget, then evaluated
on the quick official holdout (default 10 seeds x 5 scenarios = 50 episodes).
The smoke test is only an execution check; screening scores select configs for
the full 200-episode evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "tuning_results" / "a3c"

CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.010, "rollout_steps": 20},
    {"learning_rate": 2e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.005, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.005, "rollout_steps": 20},
    {"learning_rate": 5e-4, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.001, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95, "entropy_coef": 0.005, "rollout_steps": 20},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.90, "entropy_coef": 0.005, "rollout_steps": 40},
    {"learning_rate": 2e-4, "gamma": 0.98, "gae_lambda": 0.95, "entropy_coef": 0.001, "rollout_steps": 40},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 1.00, "entropy_coef": 0.001, "rollout_steps": 10},
]


def _evaluate_quick(model_path: Path, seed_count: int) -> dict[str, float]:
    cmd = [
        sys.executable,
        "-m",
        "training_pipelines.training_scripts.evaluate_a3c_rep_c",
        "--model", str(model_path),
        "--seeds", str(seed_count),
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    output = completed.stdout
    match = re.search(r"mean_cost=([0-9.,]+)", output)
    if not match:
        raise RuntimeError("Could not parse mean_cost from A3C evaluator output")
    service_match = re.search(r"mean_service_level=([0-9.]+)", output)
    return {
        "quick_mean_cost": float(match.group(1).replace(",", "")),
        "quick_mean_service_level": float(service_match.group(1)) if service_match else float("nan"),
    }


def _run(config: dict, steps: int, seed: int, workers: int, eval_seeds: int) -> dict:
    tag = f"lr{config['learning_rate']:.0e}_g{config['gamma']}_lam{config['gae_lambda']}_ent{config['entropy_coef']}_r{config['rollout_steps']}"
    run_name = f"screen_{tag}_seed{seed}"
    cmd = [
        sys.executable,
        "-m", "training_pipelines.training_scripts.train_a3c_rep_c",
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

    model_path = PROJECT_ROOT / "training_pipelines" / "models" / "a3c_rep_c_experiments" / run_name / "a3c_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"A3C checkpoint was not created: {model_path}")
    scores = _evaluate_quick(model_path, eval_seeds)
    payload = {"model": str(model_path), "config": config, "run_name": run_name, "steps": steps, "workers": workers, "eval_seeds": eval_seeds, **scores}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"quick_mean_cost={scores['quick_mean_cost']:,.2f} quick_service={scores['quick_mean_service_level']:.6f}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1 or args.eval_seeds < 1:
        raise ValueError("steps, workers and eval-seeds must be positive")
    selected = CONFIGS[: max(1, min(args.max_runs, len(CONFIGS)))]
    rows = [_run(cfg, args.steps, args.seed, args.workers, args.eval_seeds) for cfg in selected]
    rows.sort(key=lambda row: row["quick_mean_cost"])
    out = RESULTS_DIR / "screening_manifest.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== A3C SCREENING RANKING ===")
    for rank, row in enumerate(rows, 1):
        cfg = row["config"]
        print(f"{rank:>2}. cost={row['quick_mean_cost']:>10,.2f} service={row['quick_mean_service_level']:.6f} lr={cfg['learning_rate']} gamma={cfg['gamma']} lambda={cfg['gae_lambda']} ent={cfg['entropy_coef']} rollout={cfg['rollout_steps']}")
    print(f"Saved A3C screening manifest to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
