
"""Train the best TD(lambda) config from the grid search across 3 seeds
(doc's "stable across at least 3 seeds" promotion criterion, section 8.4).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.algorithms.common.env_factory import make_raw_eval_env, make_raw_training_env
from src.algorithms.td_lambda import TDLambdaConfig, train_td_lambda

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "td_lambda"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase5"

BEST_GAMMA = 0.98
BEST_LAMBDA = 0.7
BEST_ALPHA = 0.01
TOTAL_EPISODES = 5000
EVAL_EVERY = 250
EPSILON_DECAY_EPISODES = 4000  # 80% of TOTAL_EPISODES
SEEDS = (20260825, 20260826, 20260827)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        cfg = TDLambdaConfig(
            gamma=BEST_GAMMA,
            lambda_=BEST_LAMBDA,
            alpha=BEST_ALPHA,
            total_episodes=TOTAL_EPISODES,
            eval_every=EVAL_EVERY,
            epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
            seed=seed,
        )
        save_dir = MODELS_DIR / f"seed{seed}"
        result = train_td_lambda(
            lambda: make_raw_training_env(shaping=True, shaping_kwargs={"anneal_steps": TOTAL_EPISODES * 50}),
            lambda: make_raw_eval_env(),
            save_dir,
            cfg,
            verbose=False,
        )
        final_record = result["history"][-1]
        row = {
            "seed": seed,
            "best_eval_cost": result["best_eval_cost"],
            "final_eval_cost": final_record["eval_mean_cost"],
            "final_eval_service": final_record["eval_mean_service"],
        }
        results.append(row)
        print(
            f"seed={seed} best_cost={row['best_eval_cost']:>10,.1f} "
            f"final_cost={row['final_eval_cost']:>10,.1f} "
            f"final_service={row['final_eval_service']:.3f}",
            flush=True,
        )

    with open(RESULTS_DIR / "three_seed_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    costs = [r["best_eval_cost"] for r in results]
    services = [r["final_eval_service"] for r in results]
    print(f"\nMean best cost across seeds: {sum(costs)/len(costs):,.1f}")
    print(f"Std best cost across seeds: {(sum((c - sum(costs)/len(costs))**2 for c in costs)/len(costs))**0.5:,.1f}")
    print(f"Mean final service across seeds: {sum(services)/len(services):.3f}")
    best_seed = min(results, key=lambda r: r["best_eval_cost"])["seed"]
    print(f"Best seed: {best_seed}")


if __name__ == "__main__":
    main()