
"""Promote the top screening configs to a longer PPO run (Phase 6, doc 9.2:
"Promote the top two, then train to 1,500,000 to 3,000,000 transitions").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_ppo_screening import MODELS_DIR, RESULTS_DIR, train_one_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    with open(RESULTS_DIR / "ppo_screening_results.json", "r", encoding="utf-8") as handle:
        screening_results = json.load(handle)

    top_configs = screening_results[: args.top_n]
    print(f"Promoting top {len(top_configs)} configs to {args.timesteps} timesteps:")
    for row in top_configs:
        print(f"  ent_coef={row['ent_coef']} lr={row['learning_rate']} (screening best={row['best_mean_reward']:.1f})")

    final_results = []
    for row in top_configs:
        ent_coef, lr = row["ent_coef"], row["learning_rate"]
        config_id = f"promoted_ent{ent_coef}_lr{lr}"
        save_dir = MODELS_DIR / config_id
        print(f"\n=== training {config_id} ({args.timesteps} timesteps) ===", flush=True)
        result = train_one_config(ent_coef, lr, args.timesteps, save_dir)
        final_results.append(result)
        print(f"  best_mean_reward={result['best_mean_reward']:.3f}", flush=True)

    final_results.sort(key=lambda r: r["best_mean_reward"], reverse=True)
    with open(RESULTS_DIR / "ppo_promoted_results.json", "w", encoding="utf-8") as handle:
        json.dump(final_results, handle, indent=2)

    print("\n=== Final promoted results (best mean reward first) ===")
    for row in final_results:
        print(row)
    print(f"\nChampion config: {final_results[0]}")


if __name__ == "__main__":
    main()