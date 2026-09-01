

"""Grid search over TD(lambda)'s starting grid (doc section 8.3):
gamma in [0.98, 0.99], lambda in [0.7, 0.85, 0.95], alpha in [0.005, 0.01, 0.02].
"""
from __future__ import annotations

import json
from pathlib import Path

from src.algorithms.common.env_factory import make_raw_eval_env, make_raw_training_env
from src.algorithms.td_lambda import TDLambdaConfig, train_td_lambda

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "phase5"

GAMMAS = (0.98, 0.99)
LAMBDAS = (0.7, 0.85, 0.95)
ALPHAS = (0.005, 0.01, 0.02)
TOTAL_EPISODES = 2000
EVAL_EVERY = 500
EPSILON_DECAY_EPISODES = 1600  # 80% of TOTAL_EPISODES


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for gamma in GAMMAS:
        for lambda_ in LAMBDAS:
            for alpha in ALPHAS:
                cfg = TDLambdaConfig(
                    gamma=gamma,
                    lambda_=lambda_,
                    alpha=alpha,
                    total_episodes=TOTAL_EPISODES,
                    eval_every=EVAL_EVERY,
                    epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
                )
                save_dir = RESULTS_DIR / f"g{gamma}_l{lambda_}_a{alpha}"
                result = train_td_lambda(
                    lambda: make_raw_training_env(shaping=True, shaping_kwargs={"anneal_steps": TOTAL_EPISODES * 50}),
                    lambda: make_raw_eval_env(),
                    save_dir,
                    cfg,
                    verbose=False,
                )
                final_record = result["history"][-1]
                row = {
                    "gamma": gamma,
                    "lambda": lambda_,
                    "alpha": alpha,
                    "best_eval_cost": result["best_eval_cost"],
                    "final_eval_cost": final_record["eval_mean_cost"],
                    "final_eval_service": final_record["eval_mean_service"],
                }
                results.append(row)
                print(
                    f"gamma={gamma} lambda={lambda_} alpha={alpha:<6} "
                    f"best_cost={row['best_eval_cost']:>10,.1f} "
                    f"final_cost={row['final_eval_cost']:>10,.1f} "
                    f"final_service={row['final_eval_service']:.3f}"
                )

    results.sort(key=lambda r: r["best_eval_cost"])
    with open(RESULTS_DIR / "grid_search_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== Top 5 configurations by best_eval_cost ===")
    for row in results[:5]:
        print(row)


if __name__ == "__main__":
    main()