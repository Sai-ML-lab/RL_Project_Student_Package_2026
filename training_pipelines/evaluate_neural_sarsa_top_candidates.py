"""Officially evaluate the strongest Neural SARSA screening candidates.

Reads the portfolio screening manifest, selects the top-N candidates by the
screening metric, and evaluates the best checkpoint from each candidate on
the official holdout suite (40 seeds x 5 scenarios = 200 episodes by default).
No training is performed here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = TRAINING_ROOT / "eval_results" / "neural_sarsa_portfolio_screen"


def _checkpoint_for(run_dir: Path) -> Path:
    for name in ("policy_state.pt", "policy_state_final.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No SARSA checkpoint found in {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--screening", type=Path, default=RESULTS_DIR / "screening_results.json")
    args = parser.parse_args()

    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    screening_path = args.screening.resolve()
    rows = json.loads(screening_path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: float(row["best_eval_cost"]))[: args.top_n]

    print(f"Evaluating top {len(rows)} Neural SARSA candidates on official holdout ({args.seeds} seeds x 5 scenarios).", flush=True)

    results = []
    for rank, row in enumerate(rows, 1):
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (PROJECT_ROOT / run_dir).resolve()
        checkpoint = _checkpoint_for(run_dir)
        output = run_dir / "official_holdout.json"

        print(f"\n=== candidate {rank}: {row['config_id']} seed={row['seed']} ===", flush=True)
        print(f"screening_cost={float(row['best_eval_cost']):,.2f} at step={int(row['best_eval_step']):,}", flush=True)
        print(f"checkpoint={checkpoint}", flush=True)

        cmd = [
            sys.executable,
            str(TRAINING_ROOT / "evaluate_neural_sarsa_official.py"),
            "--checkpoint",
            str(checkpoint),
            "--seeds",
            str(args.seeds),
            "--output",
            str(output),
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        official = json.loads(output.read_text(encoding="utf-8"))
        overall = official["overall"]
        results.append({
            **row,
            "official_mean_cost": float(overall["mean_cost"]),
            "official_std_cost": float(overall["std_cost"]),
            "official_median_cost": float(overall["median_cost"]),
            "official_service": float(overall["mean_service_level"]),
            "official_episodes": int(overall["n_episodes"]),
            "official_output": str(output),
        })

    results.sort(key=lambda row: row["official_mean_cost"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = RESULTS_DIR / "top_candidates_official_holdout.json"
    combined.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Neural SARSA official holdout ranking ===")
    for rank, row in enumerate(results, 1):
        print(
            f"{rank:2d}. {row['config_id']} seed={row['seed']} "
            f"screen={float(row['best_eval_cost']):,.2f} "
            f"official={float(row['official_mean_cost']):,.2f} "
            f"service={float(row['official_service']):.6f}"
        )
    print(f"\nSaved combined official results to {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
