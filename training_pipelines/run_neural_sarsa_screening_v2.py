
"""Neural SARSA re-screening (iteration-3 "RL Final Run" doc, Phase 5).

Uses the FIXED Representation B features (76-dim, zero-padding + DOS bugs
corrected -- see src/features/engineered.py) and the doc's exact SARSA-A/B/C
screening configs. Does not touch the iteration-2 `models/neural_sarsa/`
checkpoints -- saves to a separate `models/neural_sarsa_v3/` tree.

Promotion rule (doc): mean cost < 140k, service >= 96%, no scenario mean
above 190k -> promote top 2 configs across 3 seeds each (see
run_neural_sarsa_promote_v2.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.algorithms.common.env_factory import make_eval_env, make_training_env
from src.algorithms.neural_sarsa import NeuralSarsaConfig, train_neural_sarsa

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "neural_sarsa_v3"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase5_v3"

SCREEN_TRANSITIONS = 300_000
SEED = 20260825

# Doc's exact SARSA-A/B/C screening configs.
CONFIGS = {
    "sarsa_a": {"learning_rate": 2e-4, "gamma": 0.99, "epsilon_end": 0.03},
    "sarsa_b": {"learning_rate": 1e-4, "gamma": 0.99, "epsilon_end": 0.05},
    "sarsa_c": {"learning_rate": 3e-4, "gamma": 0.98, "epsilon_end": 0.03},
}


def train_one_config(config_id: str, params: dict, total_transitions: int, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = NeuralSarsaConfig(
        gamma=params["gamma"],
        learning_rate=params["learning_rate"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay_transitions=int(0.8 * total_transitions),  # keep decay inside the run
        seed=SEED,
        total_transitions=total_transitions,
    )

    def env_fn():
        return make_training_env(shaping=True, shaping_kwargs={"anneal_steps": total_transitions})

    def eval_env_fn():
        return make_eval_env()

    result = train_neural_sarsa(env_fn, eval_env_fn, save_dir, cfg, verbose=True)
    return {
        "config_id": config_id,
        **params,
        "best_eval_cost": result["best_eval_cost"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-transitions", type=int, default=SCREEN_TRANSITIONS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for config_id, params in CONFIGS.items():
        save_dir = MODELS_DIR / config_id
        print(f"=== training {config_id} {params} ({args.total_transitions} transitions) ===", flush=True)
        row = train_one_config(config_id, params, args.total_transitions, save_dir)
        results.append(row)
        print(f"  best_eval_cost={row['best_eval_cost']:.1f}", flush=True)

    results.sort(key=lambda r: r["best_eval_cost"])  # lower cost first
    with open(RESULTS_DIR / "neural_sarsa_screening_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== Ranked screening results (lowest cost first) ===")
    for row in results:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())