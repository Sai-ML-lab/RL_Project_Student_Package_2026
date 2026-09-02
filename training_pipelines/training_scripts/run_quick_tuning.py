"""Hyperparameter screening for Rep-C DQN and Double DQN."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "tuning_results"

DQN_CONFIGS = [
    {"learning_rate": 1e-4, "gamma": 0.98, "target_update_interval": 1000, "exploration_final_eps": 0.02},
    {"learning_rate": 3e-4, "gamma": 0.98, "target_update_interval": 2000, "exploration_final_eps": 0.02},
    {"learning_rate": 3e-4, "gamma": 0.99, "target_update_interval": 2000, "exploration_final_eps": 0.01},
    {"learning_rate": 5e-4, "gamma": 0.99, "target_update_interval": 4000, "exploration_final_eps": 0.02},
]
DDQN_CONFIGS = list(DQN_CONFIGS)


def _best_eval_cost(run_dir: Path) -> float | None:
    path = run_dir / "eval_logs" / "evaluations.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        rewards = np.asarray(data["results"], dtype=np.float64)
    if rewards.size == 0:
        return None
    return float(-100.0 * np.max(rewards.mean(axis=1)))


def _run(kind: str, config: dict, timesteps: int, seed: int, results: list[dict]) -> None:
    module = (
        "training_pipelines.training_scripts.train_dqn_rep_c"
        if kind == "dqn"
        else "training_pipelines.training_scripts.train_double_dqn_rep_c"
    )
    tag = "_".join(f"{k}{v}" for k, v in config.items())
    run_name = f"screen_{kind}_seed{seed}_{tag}"
    cmd = [sys.executable, "-m", module, "--timesteps", str(timesteps), "--seed", str(seed), "--run-name", run_name]
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    print(f"\n=== {kind.upper()} {run_name} ===", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    run_dir = PROJECT_ROOT / "training_pipelines" / "models" / ("dqn_rep_c_experiments" if kind == "dqn" else "double_dqn_rep_c_experiments") / run_name
    best_cost = _best_eval_cost(run_dir)
    row = {"kind": kind, "seed": seed, "timesteps": timesteps, **config, "run_name": run_name, "best_quick_eval_cost": best_cost, "run_dir": str(run_dir)}
    results.append(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "screening_results.json").write_text(json.dumps(sorted(results, key=lambda r: r["best_quick_eval_cost"] if r["best_quick_eval_cost"] is not None else float("inf")), indent=2), encoding="utf-8")
    print(f"Best quick-eval cost: {best_cost}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--kind", choices=("all", "dqn", "ddqn"), default="all")
    args = parser.parse_args()
    if args.timesteps < 1 or args.max_runs < 1:
        raise ValueError("timesteps and max-runs must be positive")
    configs: list[tuple[str, dict]] = []
    if args.kind in ("all", "dqn"):
        configs.extend(("dqn", c) for c in DQN_CONFIGS)
    if args.kind in ("all", "ddqn"):
        configs.extend(("ddqn", c) for c in DDQN_CONFIGS)
    results: list[dict] = []
    for kind, config in configs[:args.max_runs]:
        _run(kind, config, args.timesteps, args.seed, results)
    ranked = sorted(results, key=lambda r: r["best_quick_eval_cost"] if r["best_quick_eval_cost"] is not None else float("inf"))
    print("\n=== QUICK-TUNING RANKING ===")
    for rank, row in enumerate(ranked, 1):
        print(f"{rank:>2}. {row['kind'].upper():4} cost={row['best_quick_eval_cost']} lr={row['learning_rate']} gamma={row['gamma']} target={row['target_update_interval']} eps={row['exploration_final_eps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
