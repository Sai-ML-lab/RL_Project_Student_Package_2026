"""Targeted A2C Representation C screening for portfolio improvement."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE = "training_pipelines.training_scripts.train_a2c_rep_c"
EVAL_MODULE = "training_pipelines.training_scripts.evaluate_a2c_rep_c"
RESULTS_DIR = PROJECT_ROOT / "tuning_results" / "a2c_rep_c"
MODEL_ROOT = PROJECT_ROOT / "training_pipelines" / "models" / "a2c_rep_c_experiments"

# Deliberately small, hypothesis-driven search. The current main A2C is already
# a strong ~114k portfolio member; these configs target lower holding/discarding
# without exploding the search budget.
CONFIGS = [
    {"learning_rate": 5e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.001, "n_steps": 64, "vf_coef": 0.5, "no_shaping": False},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.001, "n_steps": 64, "vf_coef": 0.5, "no_shaping": False},
    {"learning_rate": 5e-4, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.0, "n_steps": 64, "vf_coef": 0.5, "no_shaping": False},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "n_steps": 128, "vf_coef": 0.5, "no_shaping": False},
    {"learning_rate": 5e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "n_steps": 64, "vf_coef": 0.25, "no_shaping": True},
    {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "n_steps": 128, "vf_coef": 0.25, "no_shaping": True},
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
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"A2C Rep-C evaluator failed with exit code {completed.returncode}")

    cost_match = re.search(r"mean_cost=([0-9.,]+)", completed.stdout)
    service_match = re.search(r"mean_service_level=([0-9.]+)", completed.stdout)
    if cost_match is None:
        raise RuntimeError("Could not parse A2C Rep-C mean_cost")
    return {
        "mean_cost": float(cost_match.group(1).replace(",", "")),
        "mean_service_level": float(service_match.group(1)) if service_match else float("nan"),
    }


def _run(config: dict, steps: int, seed: int, n_envs: int, eval_seeds: int) -> dict:
    shaping_tag = "noshape" if config["no_shaping"] else "shape"
    tag = (
        f"lr{config['learning_rate']:.0e}_g{config['gamma']}_"
        f"lam{config['gae_lambda']}_ent{config['ent_coef']}_"
        f"n{config['n_steps']}_vf{config['vf_coef']}_{shaping_tag}"
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
        "--learning-rate", str(config["learning_rate"]),
        "--gamma", str(config["gamma"]),
        "--gae-lambda", str(config["gae_lambda"]),
        "--ent-coef", str(config["ent_coef"]),
        "--n-steps", str(config["n_steps"]),
        "--vf-coef", str(config["vf_coef"]),
    ]
    if config["no_shaping"]:
        cmd.append("--no-shaping")

    print(f"\n=== A2C Rep-C {run_name} ===", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    model_path = MODEL_ROOT / run_name / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No A2C Rep-C checkpoint found: {model_path}")

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

    rows = [_run(config, args.steps, args.seed, args.n_envs, args.eval_seeds) for config in CONFIGS[:args.max_runs]]
    rows.sort(key=lambda row: row["mean_cost"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "screening_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== A2C REP-C SCREENING RANKING ===")
    for rank, row in enumerate(rows, 1):
        c = row["config"]
        print(
            f"{rank:>2}. cost={row['mean_cost']:>10,.2f} "
            f"service={row['mean_service_level']:.6f} "
            f"lr={c['learning_rate']} gamma={c['gamma']} "
            f"lambda={c['gae_lambda']} ent={c['ent_coef']} "
            f"n_steps={c['n_steps']} vf={c['vf_coef']} "
            f"shaping={not c['no_shaping']}"
        )
    print(f"Saved A2C Rep-C screening manifest to {RESULTS_DIR / 'screening_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
