"""Official-holdout evaluator for a saved Neural SARSA checkpoint.

The checkpoint is loaded with the same QNetwork architecture used by the
training algorithm. Inference uses canonical Representation B (76-D) and the
full 1331-action joint-action catalog, then calls the shared official
`evaluation.py` harness for comparable 200-episode results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES, evaluate_policy, summarise_overall
from src.algorithms.common.checkpoint import load_checkpoint
from src.algorithms.common.networks import QNetwork
from src.environment.action_codec import decode_joint_action
from src.features.engineered import REPRESENTATION_B_DIM, flatten_observation_representation_b


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("policy_state.pt", "policy_state_final.pt"):
            candidate = path / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No SARSA checkpoint found at {path}. Expected policy_state.pt or policy_state_final.pt."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    obs_dim = int(payload["obs_dim"])
    n_actions = int(payload["n_actions"])
    hidden_sizes = tuple(int(x) for x in payload["hidden_sizes"])

    if obs_dim != REPRESENTATION_B_DIM:
        raise ValueError(f"Checkpoint obs_dim={obs_dim}; expected Representation B {REPRESENTATION_B_DIM}")

    model = QNetwork(obs_dim, n_actions, hidden_sizes)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def policy(observation):
        features = flatten_observation_representation_b(observation)
        if features.shape != (REPRESENTATION_B_DIM,):
            raise ValueError(f"Unexpected Representation B shape: {features.shape}")
        with torch.no_grad():
            q_values = model(torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0))
        action_index = int(torch.argmax(q_values, dim=-1).item())
        quantities = decode_joint_action(action_index)
        values = [int(x) for x in quantities]
        if len(values) != 3 or any(x < 0 or x > 100 or x % 10 for x in values):
            raise ValueError(f"Unexpected decoded action: {values}")
        return values

    seeds = HOLDOUT_SEEDS[: max(1, min(args.seeds, len(HOLDOUT_SEEDS)))]
    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=seeds,
        scenario_modes=SCENARIO_MODES,
        domain_randomization=True,
        progress=True,
    )
    overall = summarise_overall(per_episode)

    print("\n=== Neural SARSA official holdout ===")
    print(f"checkpoint={checkpoint_path}")
    print(f"episodes={overall['n_episodes']}")
    print(f"mean_cost={overall['mean_cost']:,.2f}")
    print(f"std_cost={overall['std_cost']:,.2f}")
    print(f"median_cost={overall['median_cost']:,.2f}")
    print(f"mean_service_level={overall['mean_service_level']:.6f}")
    print("\nScenario summary:")
    print(scenario_summary.to_string(index=False))

    result = {
        "checkpoint": str(checkpoint_path),
        "overall": {str(k): (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in overall.items()},
        "scenario_summary": scenario_summary.to_dict(orient="records"),
    }
    output_path = args.output or (checkpoint_path.parent / "official_holdout.json")
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved official evaluation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
