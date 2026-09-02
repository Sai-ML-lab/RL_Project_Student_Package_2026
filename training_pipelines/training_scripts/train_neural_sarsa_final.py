"""Reproduce the retained final Neural Network SARSA submission.

Configuration matches the frozen public SARSA artifact:
- Representation B (76-D)
- true SARSA target with the actual next behavior action
- learning rate 3e-4, gamma .98, epsilon 1.0 -> .03
- 500,000 transitions, seed 20260825
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training_pipelines"
for path in (PROJECT_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.algorithms.common.env_factory import make_eval_env, make_training_env
from src.algorithms.neural_sarsa import NeuralSarsaConfig, train_neural_sarsa

MODELS_DIR = TRAINING_ROOT / "models" / "neural_sarsa_final"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the retained final Neural SARSA configuration")
    parser.add_argument("--total-transitions", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--run-name", type=str, default="final_neural_sarsa_500k")
    args = parser.parse_args()

    if args.total_transitions < 1:
        raise ValueError("--total-transitions must be >= 1")

    save_dir = MODELS_DIR / args.run_name
    cfg = NeuralSarsaConfig(
        gamma=0.98,
        learning_rate=3e-4,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_transitions=int(0.8 * args.total_transitions),
        batch_size=128,
        buffer_size=50_000,
        learning_starts=5_000,
        train_frequency=1,
        gradient_steps=1,
        target_update_interval=2_000,
        seed=args.seed,
        total_transitions=args.total_transitions,
        eval_every=25_000,
        eval_seeds=tuple(range(900, 910)),
    )

    def env_fn():
        return make_training_env(
            shaping=True,
            shaping_kwargs={"anneal_steps": args.total_transitions},
        )

    def eval_env_fn():
        return make_eval_env()

    print(f"Training final Neural SARSA: {cfg}", flush=True)
    result = train_neural_sarsa(env_fn, eval_env_fn, save_dir, cfg, verbose=True)
    print(f"Best internal evaluation cost: {result['best_eval_cost']:,.2f}", flush=True)
    print(f"Final checkpoint: {save_dir / 'policy_state_final.pt'}", flush=True)
    print(f"Best checkpoint: {save_dir / 'policy_state.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
