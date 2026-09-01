"""Neural SARSA + Representation B candidate experiment.

The underlying Neural SARSA algorithm is kept unchanged.  This launcher makes
Representation B explicit, supports controlled hyperparameters/seeds, and
evaluates the saved best checkpoint on the authoritative 200-episode local
holdout (40 seeds x 5 scenario modes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

for _path in (REPO_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from src.algorithms.common.env_factory import make_eval_env, make_training_env
from src.algorithms.common.checkpoint import load_checkpoint
from src.algorithms.common.networks import QNetwork
from src.algorithms.neural_sarsa import NeuralSarsaConfig, train_neural_sarsa
from src.features.engineered import REPRESENTATION_B_DIM, flatten_observation_representation_b

MODELS_DIR = PROJECT_ROOT / "models" / "neural_sarsa_rep_b_experiments"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "neural_sarsa_rep_b_experiments"

DEFAULT_SEED = 20260826


def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = load_checkpoint(checkpoint_path)

    if int(ckpt["obs_dim"]) != REPRESENTATION_B_DIM:
        raise ValueError(
            f"Checkpoint obs_dim={ckpt['obs_dim']} but Representation B "
            f"requires {REPRESENTATION_B_DIM}."
        )

    q_net = QNetwork(
        int(ckpt["obs_dim"]),
        int(ckpt["n_actions"]),
        ckpt["hidden_sizes"],
    )
    q_net.load_state_dict(ckpt["model_state"])
    q_net.eval()

    def policy(observation):
        features = flatten_observation_representation_b(observation)
        with torch.no_grad():
            q_values = q_net(
                torch.from_numpy(
                    np.asarray(features, dtype=np.float32)
                ).unsqueeze(0)
            )
        joint_index = int(torch.argmax(q_values, dim=-1).item())

        a3 = joint_index % 11
        rem = joint_index // 11
        a2 = rem % 11
        a1 = rem // 11
        return [a1 * 10, a2 * 10, a3 * 10]

    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=HOLDOUT_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=PROJECT_ROOT / "assigned_config.json",
        domain_randomization=True,
        progress=False,
    )
    overall = summarise_overall(per_episode)

    return {
        "mean_cost": overall["mean_cost"],
        "std_cost": overall["std_cost"],
        "median_cost": overall["median_cost"],
        "mean_service_level": overall["mean_service_level"],
        "mean_call_ms": overall["mean_call_ms"],
        "n_episodes": overall["n_episodes"],
        "scenario_summary": scenario_summary.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Neural SARSA + Representation B experiment."
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.8)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError("--timesteps must be >= 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0 <= args.epsilon_end <= 1:
        raise ValueError("--epsilon-end must be in [0, 1]")
    if not 0 < args.epsilon_decay_fraction <= 1:
        raise ValueError("--epsilon-decay-fraction must be in (0, 1]")

    run_name = args.run_name or (
        f"sarsa_rep_b_{args.timesteps // 1000}k"
        f"_lr{args.learning_rate:.0e}"
        f"_g{args.gamma}"
        f"_e{args.epsilon_end}"
    )
    save_dir = MODELS_DIR / f"{run_name}_seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Neural SARSA + Representation B ===", flush=True)
    print(
        f"obs_dim={REPRESENTATION_B_DIM} "
        f"timesteps={args.timesteps:,} "
        f"lr={args.learning_rate} "
        f"gamma={args.gamma} "
        f"epsilon_end={args.epsilon_end} "
        f"seed={args.seed}",
        flush=True,
    )

    cfg = NeuralSarsaConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_end=args.epsilon_end,
        epsilon_decay_transitions=max(
            1,
            int(args.epsilon_decay_fraction * args.timesteps),
        ),
        seed=args.seed,
        total_transitions=args.timesteps,
    )

    # The custom SARSA env factory already exposes:
    # IndustrialInventoryEnv -> ShapedReward -> JointActionWrapper
    # -> EngineeredObsWrapper (76-D).
    def env_fn():
        return make_training_env(
            shaping=True,
            shaping_kwargs={"anneal_steps": args.timesteps},
        )

    def eval_env_fn():
        return make_eval_env()

    result = train_neural_sarsa(
        env_fn,
        eval_env_fn,
        save_dir,
        cfg,
        verbose=True,
    )

    best_checkpoint = save_dir / "policy_state.pt"
    final_checkpoint = save_dir / "policy_state_final.pt"

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Expected best SARSA checkpoint: {best_checkpoint}"
        )

    evaluation = evaluate_checkpoint(best_checkpoint)

    output = {
        "technique": "Neural Network SARSA",
        "representation": "B",
        "observation_dim": REPRESENTATION_B_DIM,
        "run_id": save_dir.name,
        "seed": args.seed,
        "total_transitions": args.timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "best_internal_eval_cost": result["best_eval_cost"],
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "official_local_holdout": evaluation,
    }

    result_path = RESULTS_DIR / f"{save_dir.name}.json"
    result_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n=== SARSA Rep-B experiment complete ===", flush=True)
    print(f"Best checkpoint : {best_checkpoint}", flush=True)
    print(f"Mean cost       : {evaluation['mean_cost']:,.2f}", flush=True)
    print(f"Std cost        : {evaluation['std_cost']:,.2f}", flush=True)
    print(f"Median cost     : {evaluation['median_cost']:,.2f}", flush=True)
    print(f"Service         : {evaluation['mean_service_level']:.6f}", flush=True)
    print(f"Result file     : {result_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
