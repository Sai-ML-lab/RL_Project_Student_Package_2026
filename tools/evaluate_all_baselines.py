from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Make repository root importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (
    HOLDOUT_SEEDS,
    SCENARIO_MODES,
    evaluate_policy,
    summarise_overall,
)


SUBMISSIONS = [
    ("A2C", ROOT / "submissions" / "a2c" / "policy.py"),
    ("DQN", ROOT / "submissions" / "dqn" / "policy.py"),
    ("PPO", ROOT / "submissions" / "ppo" / "policy.py"),
    (
        "Neural SARSA",
        ROOT / "submissions" / "neural_sarsa" / "policy.py",
    ),
    (
        "TD(lambda)",
        ROOT / "submissions" / "td_lambda" / "policy.py",
    ),
]


def load_policy(name: str, path: Path):
    """Load run_policy from a submission policy.py."""

    module_name = (
        "baseline_"
        + name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load policy module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run_policy"):
        raise AttributeError(
            f"{path} does not define run_policy()."
        )

    return module.run_policy


def main() -> None:

    print("=" * 80)
    print("RL PROJECT - CURRENT MODEL BASELINE")
    print("=" * 80)
    print(
        f"Seeds: {len(HOLDOUT_SEEDS)} "
        f"({HOLDOUT_SEEDS[0]}..{HOLDOUT_SEEDS[-1]})"
    )
    print(
        f"Scenarios: {', '.join(SCENARIO_MODES)}"
    )
    print(
        f"Episodes per model: "
        f"{len(HOLDOUT_SEEDS) * len(SCENARIO_MODES)}"
    )
    print("=" * 80)

    overall_rows: list[dict] = []
    scenario_rows: list[pd.DataFrame] = []

    for model_name, policy_path in SUBMISSIONS:

        print(
            f"\n{'=' * 20} {model_name} {'=' * 20}"
        )

        if not policy_path.exists():
            raise FileNotFoundError(
                f"Policy file not found: {policy_path}"
            )

        policy = load_policy(
            model_name,
            policy_path,
        )

        print(
            f"Running {len(HOLDOUT_SEEDS) * len(SCENARIO_MODES)} "
            "episodes..."
        )

        per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=HOLDOUT_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=ROOT / "training_pipelines" / "assigned_config.json",
        domain_randomization=True,
        progress=True,
    )
        overall = summarise_overall(
            per_episode
        )

        overall["model"] = model_name

        overall_rows.append(overall)

        scenario_summary = scenario_summary.copy()
        scenario_summary.insert(
            0,
            "model",
            model_name,
        )

        scenario_rows.append(
            scenario_summary
        )

        print(
            f"\n{model_name} overall:"
        )

        print(
            f"  mean cost        : "
            f"{overall['mean_cost']:,.2f}"
        )
        print(
            f"  std cost         : "
            f"{overall['std_cost']:,.2f}"
        )
        print(
            f"  median cost      : "
            f"{overall['median_cost']:,.2f}"
        )
        print(
            f"  mean service     : "
            f"{overall['mean_service_level']:.6f}"
        )
        print(
            f"  mean policy ms   : "
            f"{overall['mean_call_ms']:.4f}"
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    scenario_df = pd.concat(
        scenario_rows,
        ignore_index=True,
    )

    overall_df = overall_df[
        [
            "model",
            "mean_cost",
            "std_cost",
            "median_cost",
            "mean_service_level",
            "mean_call_ms",
            "n_episodes",
        ]
    ]

    overall_df = overall_df.sort_values(
        "mean_cost"
    ).reset_index(drop=True)

    output_dir = (
        ROOT
        / "eval_results"
        / "model_baseline_2026_09_01"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        output_dir
        / "overall_summary.csv"
    )

    scenario_path = (
        output_dir
        / "scenario_summary.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False,
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("FINAL BASELINE RANKING")
    print("=" * 80)

    display_df = overall_df.copy()

    display_df["mean_cost"] = display_df[
        "mean_cost"
    ].map(lambda x: f"{x:,.2f}")

    display_df["median_cost"] = display_df[
        "median_cost"
    ].map(lambda x: f"{x:,.2f}")

    display_df["std_cost"] = display_df[
        "std_cost"
    ].map(lambda x: f"{x:,.2f}")

    display_df["mean_service_level"] = display_df[
        "mean_service_level"
    ].map(lambda x: f"{x:.6f}")

    print(
        display_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(overall_path)
    print(scenario_path)


if __name__ == "__main__":
    main()