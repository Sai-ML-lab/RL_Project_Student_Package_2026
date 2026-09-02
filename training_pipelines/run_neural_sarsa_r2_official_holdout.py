"""Official 200-episode holdout evaluation for the final Neural SARSA r2 candidate.

This evaluates the best r2 checkpoint on the exact leaderboard-style holdout:
40 seeds (900-939) x 5 scenario modes, using the unshaped environment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make both repo root and training_pipelines importable when launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation import HOLDOUT_SEEDS, SCENARIO_MODES  # noqa: E402
from src.algorithms.common.checkpoint import load_checkpoint  # noqa: E402
from src.algorithms.common.env_factory import make_eval_env  # noqa: E402
from src.algorithms.common.networks import QNetwork  # noqa: E402


CANDIDATE_DIR = (
    TRAINING_ROOT
    / "models"
    / "neural_sarsa_final_refinement"
    / "r2_lr1.5e-4_g0.99_e0.01_seed20260905"
)
CHECKPOINT_PATH = CANDIDATE_DIR / "policy_state.pt"
RESULTS_DIR = TRAINING_ROOT / "eval_results" / "neural_sarsa_final_refinement"
OUTPUT_CSV = RESULTS_DIR / "r2_official_holdout.csv"
OUTPUT_JSON = RESULTS_DIR / "r2_official_holdout_summary.json"


def _scalar_capacity_utilisation(obs: np.ndarray) -> float:
    """Return capacity utilisation only when it is present as a scalar-like value.

    The SARSA evaluation environment uses the 76-D engineered feature vector, not
    the raw observation dict. Capacity utilisation is already included inside
    that representation, so this metric is optional and should not block the
    official cost evaluation.
    """
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    # EngineeredObsWrapper's documented Representation-B layout does include a
    # dedicated capacity feature, but its exact position is intentionally not
    # used for the official score. Return NaN here rather than guessing an index.
    return float("nan") if arr.size else float("nan")


def evaluate_checkpoint() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run the final SARSA refinement first and confirm r2 completed."
        )

    payload = load_checkpoint(CHECKPOINT_PATH)
    obs_dim = int(payload["obs_dim"])
    n_actions = int(payload["n_actions"])
    hidden_sizes = tuple(int(x) for x in payload["hidden_sizes"])

    q_net = QNetwork(obs_dim, n_actions, hidden_sizes)
    q_net.load_state_dict(payload["model_state"])
    q_net.eval()

    rows: list[dict[str, float | int | str]] = []

    print("=== Neural SARSA r2 official holdout ===", flush=True)
    print(f"Checkpoint : {CHECKPOINT_PATH}", flush=True)
    print(f"Episodes   : {len(HOLDOUT_SEEDS) * len(SCENARIO_MODES)}", flush=True)
    print(f"Obs dim    : {obs_dim}", flush=True)
    print(f"Actions    : {n_actions}", flush=True)
    print(f"Scenarios  : {SCENARIO_MODES}", flush=True)
    print(f"Seeds      : {HOLDOUT_SEEDS[0]}-{HOLDOUT_SEEDS[-1]}", flush=True)

    for scenario_mode in SCENARIO_MODES:
        env = make_eval_env(
            scenario_mode=scenario_mode,
            domain_randomization=True,
        )
        try:
            for seed in HOLDOUT_SEEDS:
                obs, reset_info = env.reset(seed=int(seed))
                obs = np.asarray(obs, dtype=np.float32)
                done = False
                total_cost = 0.0
                holding = stockout = ordering = discarding = 0.0
                total_demand = total_fulfilled = 0
                capacity_sum = 0.0
                steps = 0

                while not done:
                    with torch.no_grad():
                        q_values = q_net(
                            torch.from_numpy(obs).unsqueeze(0)
                        )
                        action = int(torch.argmax(q_values, dim=-1).item())

                    obs, _reward, terminated, truncated, info = env.step(action)
                    obs = np.asarray(obs, dtype=np.float32)
                    total_cost += float(info["costs"]["daily_total"])
                    holding += float(info["costs"]["holding"])
                    stockout += float(info["costs"]["stockout"])
                    ordering += float(info["costs"]["ordering"])
                    discarding += float(info["costs"]["discarding"])
                    total_demand += int(np.sum(info["demand"]))
                    total_fulfilled += int(np.sum(info["fulfilled_demand"]))
                    capacity_value = _scalar_capacity_utilisation(obs)
                    if np.isfinite(capacity_value):
                        capacity_sum += capacity_value
                    steps += 1
                    done = bool(terminated or truncated)

                service = total_fulfilled / max(total_demand, 1)
                avg_capacity = (
                    capacity_sum / max(steps, 1)
                    if capacity_sum != 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "scenario_mode": scenario_mode,
                        "seed": int(seed),
                        "episode_cost": total_cost,
                        "holding_cost": holding,
                        "stockout_cost": stockout,
                        "ordering_cost": ordering,
                        "discarding_cost": discarding,
                        "service_level": service,
                        "total_demand": total_demand,
                        "total_fulfilled": total_fulfilled,
                        "avg_capacity_utilisation": avg_capacity,
                        "steps": steps,
                        "variant_id": reset_info.get("variant_id"),
                        "config_fingerprint": reset_info.get("config_fingerprint"),
                    }
                )
        finally:
            env.close()

    per_episode = pd.DataFrame(rows)
    summary = (
        per_episode.groupby("scenario_mode", as_index=False)
        .agg(
            mean_cost=("episode_cost", "mean"),
            std_cost=("episode_cost", "std"),
            median_cost=("episode_cost", "median"),
            p90_cost=("episode_cost", lambda s: float(np.percentile(s, 90))),
            mean_service_level=("service_level", "mean"),
            mean_holding=("holding_cost", "mean"),
            mean_stockout=("stockout_cost", "mean"),
            mean_ordering=("ordering_cost", "mean"),
            mean_discarding=("discarding_cost", "mean"),
        )
    )

    overall = {
        "mean_cost": float(per_episode["episode_cost"].mean()),
        "std_cost": float(per_episode["episode_cost"].std()),
        "median_cost": float(per_episode["episode_cost"].median()),
        "p90_cost": float(np.percentile(per_episode["episode_cost"], 90)),
        "mean_service_level": float(per_episode["service_level"].mean()),
        "mean_holding": float(per_episode["holding_cost"].mean()),
        "mean_stockout": float(per_episode["stockout_cost"].mean()),
        "mean_ordering": float(per_episode["ordering_cost"].mean()),
        "mean_discarding": float(per_episode["discarding_cost"].mean()),
        "n_episodes": int(len(per_episode)),
    }

    return per_episode, summary, overall


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_episode, summary, overall = evaluate_checkpoint()
    per_episode.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(
        json.dumps(
            {"overall": overall, "scenario_summary": summary.to_dict(orient="records")},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Scenario summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== Overall official holdout ===", flush=True)
    print(f"Mean cost            : {overall['mean_cost']:,.2f}", flush=True)
    print(f"Std cost             : {overall['std_cost']:,.2f}", flush=True)
    print(f"Median cost          : {overall['median_cost']:,.2f}", flush=True)
    print(f"P90 cost             : {overall['p90_cost']:,.2f}", flush=True)
    print(f"Mean service level   : {overall['mean_service_level']:.6f}", flush=True)
    print(f"Mean holding cost    : {overall['mean_holding']:,.2f}", flush=True)
    print(f"Mean stockout cost   : {overall['mean_stockout']:,.2f}", flush=True)
    print(f"Mean ordering cost   : {overall['mean_ordering']:,.2f}", flush=True)
    print(f"Mean discarding cost : {overall['mean_discarding']:,.2f}", flush=True)
    print(f"Episodes             : {overall['n_episodes']}", flush=True)
    print(f"\nSaved per-episode results to {OUTPUT_CSV}", flush=True)
    print(f"Saved summary to {OUTPUT_JSON}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
