"""Final targeted Neural SARSA refinement.

Keeps the validated 500k s2 recipe and tests only a small service-oriented
perturbation of exploration/discount settings. All runs are isolated from
protected submissions. Screening is followed by official holdout evaluation
of the best checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.common.env_factory import make_eval_env, make_training_env
from src.algorithms.neural_sarsa import NeuralSarsaConfig, train_neural_sarsa

MODELS_DIR = TRAINING_ROOT / "models" / "neural_sarsa_final_refinement"
RESULTS_DIR = TRAINING_ROOT / "eval_results" / "neural_sarsa_final_refinement"
DEFAULT_TRANSITIONS = 500_000
DEFAULT_SEED = 20260905

CONFIGS = {
    "r1_lr2e-4_g0.99_e0.01": {
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "epsilon_end": 0.01,
        "batch_size": 128,
        "buffer_size": 100_000,
        "target_update_interval": 2_000,
    },
    "r2_lr1.5e-4_g0.99_e0.01": {
        "learning_rate": 1.5e-4,
        "gamma": 0.99,
        "epsilon_end": 0.01,
        "batch_size": 128,
        "buffer_size": 100_000,
        "target_update_interval": 2_000,
    },
    "r3_lr2e-4_g0.995_e0.01": {
        "learning_rate": 2e-4,
        "gamma": 0.995,
        "epsilon_end": 0.01,
        "batch_size": 128,
        "buffer_size": 100_000,
        "target_update_interval": 2_000,
    },
}


def train_one(config_id: str, params: dict[str, float | int], total_transitions: int, seed: int) -> dict:
    save_dir = MODELS_DIR / f"{config_id}_seed{seed}"
    cfg = NeuralSarsaConfig(
        gamma=float(params["gamma"]),
        learning_rate=float(params["learning_rate"]),
        epsilon_end=float(params["epsilon_end"]),
        batch_size=int(params["batch_size"]),
        buffer_size=int(params["buffer_size"]),
        target_update_interval=int(params["target_update_interval"]),
        epsilon_decay_transitions=int(0.85 * total_transitions),
        seed=seed,
        total_transitions=total_transitions,
        eval_every=25_000,
        eval_seeds=tuple(range(900, 910)),
    )

    def env_fn():
        return make_training_env(shaping=True, shaping_kwargs={"anneal_steps": total_transitions})

    def eval_env_fn():
        return make_eval_env()

    result = train_neural_sarsa(env_fn, eval_env_fn, save_dir, cfg, verbose=True)
    best = min(result["history"], key=lambda row: row["eval_mean_cost"])
    return {
        "config_id": config_id,
        "seed": seed,
        **params,
        "total_transitions": total_transitions,
        "best_eval_cost": float(result["best_eval_cost"]),
        "best_eval_service": float(best["eval_mean_service"]),
        "best_eval_step": int(best["step"]),
        "final_eval_cost": float(result["history"][-1]["eval_mean_cost"]),
        "run_dir": str(save_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-transitions", type=int, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for config_id, params in CONFIGS.items():
        print(f"\n=== {config_id} ===", flush=True)
        print(params, flush=True)
        row = train_one(config_id, params, args.total_transitions, args.seed)
        results.append(row)
        print(
            f"BEST screening cost={row['best_eval_cost']:,.1f} "
            f"service={row['best_eval_service']:.4f} "
            f"at_step={row['best_eval_step']:,}",
            flush=True,
        )

    results.sort(key=lambda r: r["best_eval_cost"])
    output = RESULTS_DIR / "refinement_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Neural SARSA final refinement ranking ===")
    for rank, row in enumerate(results, 1):
        print(
            f"{rank:2d}. {row['config_id']} "
            f"cost={row['best_eval_cost']:,.2f} "
            f"service={row['best_eval_service']:.6f} "
            f"step={row['best_eval_step']:,}"
        )
    print(f"\nSaved refinement results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
