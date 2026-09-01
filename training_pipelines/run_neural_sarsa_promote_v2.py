
"""Promote the top 2 Neural SARSA v3 screening configs across 3 seeds each
(iteration-3 doc Phase 5 promotion rule / Phase 10 Round 3). Reads
eval_results/phase5_v3/neural_sarsa_screening_results.json (produced by
run_neural_sarsa_screening_v2.py) and re-trains the top-2 configs at a
longer budget, 3 seeds each (6 runs total).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_neural_sarsa_screening_v2 import CONFIGS, train_one_config

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase5_v3"
MODELS_DIR = PROJECT_ROOT / "models" / "neural_sarsa_v3"

PROMOTE_TRANSITIONS = 500_000
SEEDS = (20260825, 20260826, 20260827)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--total-transitions", type=int, default=PROMOTE_TRANSITIONS)
    args = parser.parse_args()

    screening_path = RESULTS_DIR / "neural_sarsa_screening_results.json"
    screening_results = json.loads(screening_path.read_text(encoding="utf-8"))
    top_configs = screening_results[: args.top_n]  # already sorted lowest-cost-first
    print(f"Promoting top {args.top_n} configs: {[c['config_id'] for c in top_configs]}", flush=True)

    results = []
    for entry in top_configs:
        config_id = entry["config_id"]
        params = {k: entry[k] for k in ("learning_rate", "gamma", "epsilon_end")}
        for seed in SEEDS:
            run_id = f"promoted_{config_id}_seed{seed}"
            save_dir = MODELS_DIR / run_id
            print(f"=== {run_id} {params} ({args.total_transitions} transitions) ===", flush=True)

            # train_one_config always uses SEED=20260825 internally for env/torch
            # seeding via NeuralSarsaConfig.seed -- override per-seed here.
            import run_neural_sarsa_screening_v2 as _mod

            original_seed = _mod.SEED
            _mod.SEED = seed
            try:
                row = train_one_config(config_id, params, args.total_transitions, save_dir)
            finally:
                _mod.SEED = original_seed
            row["seed"] = seed
            row["run_id"] = run_id
            results.append(row)
            print(f"  best_eval_cost={row['best_eval_cost']:.1f}", flush=True)

    with open(RESULTS_DIR / "neural_sarsa_promoted_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== Promoted results (all runs) ===")
    for row in sorted(results, key=lambda r: r["best_eval_cost"]):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())