"""Train a joint-action Dueling Double DQN using Representation B.

This experiment keeps the full 1331-action joint decision space and changes
the value-function architecture to a dueling network.

Training reward is shaped.
Official holdout evaluation uses the unshaped environment reward/cost.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

for path in (
    REPO_ROOT,
    PROJECT_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from evaluation import (  # noqa: E402
    HOLDOUT_SEEDS,
    SCENARIO_MODES,
    evaluate_policy,
    summarise_overall,
)

from industrial_inventory_env import IndustrialInventoryEnv  # noqa: E402

from src.algorithms.dueling_double_dqn import (  # noqa: E402
    DuelingQNetwork,
    ReplayBuffer,
    double_dqn_update,
    linear_epsilon,
    select_action,
)

from src.environment.action_codec import (  # noqa: E402
    indices_to_quantities,
    joint_index_to_multidiscrete,
)

from src.features.engineered import (  # noqa: E402
    REPRESENTATION_B_DIM,
    flatten_observation_representation_b,
)

from training_utils.action_wrapper import FlattenAction  # noqa: E402
from training_utils.env_factory import load_assigned_config  # noqa: E402
from training_utils.reward_shaping import ShapedReward  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "dueling_double_dqn_experiments"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "eval_results"
    / "dueling_double_dqn_experiments"
)

N_PRODUCTS = 3
N_ACTIONS = 11 ** N_PRODUCTS

HIDDEN_SIZES = (256, 256)


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def make_train_env(
    config: dict,
    seed: int,
    anneal_steps: int,
):
    """Build the shaped training environment."""

    env = IndustrialInventoryEnv(
        student_config=config,
        scenario_mode="random",
        domain_randomization=True,
    )

    env = ShapedReward(
        env,
        anneal_steps=max(
            1,
            int(anneal_steps),
        ),
    )

    # Representation B:
    # raw 38 + engineered 38 = 76.
    from src.features.engineered import EngineeredObsWrapper

    env = EngineeredObsWrapper(env)

    # DQN-family methods use Discrete(1331).
    env = FlattenAction(env)

    env.reset(seed=seed)

    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    args,
) -> tuple[DuelingQNetwork, dict]:
    """Train the Dueling Double-DQN model."""

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(
        args.seed
    )

    # We deliberately use CPU for this small MLP workload.
    device = torch.device(
        args.device
    )

    config = load_assigned_config(
        PROJECT_ROOT
        / "assigned_config.json"
    )

    env = make_train_env(
        config=config,
        seed=args.seed,
        anneal_steps=args.timesteps,
    )

    obs_dim = REPRESENTATION_B_DIM

    online = DuelingQNetwork(
        obs_dim=obs_dim,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
    ).to(device)

    target = DuelingQNetwork(
        obs_dim=obs_dim,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
    ).to(device)

    target.load_state_dict(
        online.state_dict()
    )

    target.eval()

    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=args.learning_rate,
    )

    replay = ReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
    )

    obs, _ = env.reset(
        seed=args.seed
    )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    updates = 0
    last_loss = float("nan")

    started_at = time.perf_counter()

    heartbeat_every = max(
        1,
        args.timesteps // 20,
    )

    for step in range(
        1,
        args.timesteps + 1,
    ):
        epsilon = linear_epsilon(
            step=step,
            total_steps=args.timesteps,
            initial=1.0,
            final=args.exploration_final_eps,
            fraction=args.exploration_fraction,
        )

        action = select_action(
            q_net=online,
            obs=obs,
            epsilon=epsilon,
            rng=rng,
            n_actions=N_ACTIONS,
            device=device,
        )

        next_obs, reward, terminated, truncated, _info = env.step(
            action
        )

        next_obs = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        done = bool(
            terminated
            or truncated
        )

        replay.add(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=done,
        )

        if done:
            obs, _ = env.reset()
            obs = np.asarray(
                obs,
                dtype=np.float32,
            )
        else:
            obs = next_obs

        # Start gradient updates after the replay buffer has enough data.
        if (
            step >= args.learning_starts
            and step % args.train_frequency == 0
            and replay.size >= args.batch_size
        ):
            for _ in range(
                args.gradient_steps
            ):
                batch = replay.sample(
                    args.batch_size,
                    rng,
                )

                last_loss = double_dqn_update(
                    online=online,
                    target=target,
                    optimizer=optimizer,
                    batch=batch,
                    gamma=args.gamma,
                    device=device,
                    max_grad_norm=args.max_grad_norm,
                )

                updates += 1

                if (
                    updates
                    % args.target_update_interval
                    == 0
                ):
                    target.load_state_dict(
                        online.state_dict()
                    )

        if (
            step % heartbeat_every == 0
            or step == args.timesteps
        ):
            elapsed = (
                time.perf_counter()
                - started_at
            )

            fps = (
                step
                / max(
                    elapsed,
                    1e-9,
                )
            )

            print(
                f"step={step:,}/{args.timesteps:,} "
                f"epsilon={epsilon:.3f} "
                f"loss={last_loss:.5f} "
                f"updates={updates:,} "
                f"fps={fps:.1f}",
                flush=True,
            )

    env.close()

    online = online.cpu()

    metadata = {
        "obs_dim": obs_dim,
        "n_actions": N_ACTIONS,
        "hidden_sizes": list(HIDDEN_SIZES),
        "updates": updates,
    }

    return online, metadata


# ---------------------------------------------------------------------------
# Official holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: DuelingQNetwork,
) -> dict:
    """Run the project's official 200-episode holdout."""

    model.eval()

    config = load_assigned_config(
        PROJECT_ROOT
        / "assigned_config.json"
    )

    def policy(
        observation,
    ):
        features = flatten_observation_representation_b(
            observation
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        with torch.no_grad():
            q_values = model(
                torch.from_numpy(
                    features
                ).unsqueeze(0)
            )

            joint_index = int(
                q_values.argmax(
                    dim=1
                ).item()
            )

        action_indices = (
            joint_index_to_multidiscrete(
                joint_index
            )
        )

        return indices_to_quantities(
            action_indices
        )

    per_episode, scenario_summary = evaluate_policy(
        policy,
        seeds=HOLDOUT_SEEDS,
        scenario_modes=SCENARIO_MODES,
        config=config,
        domain_randomization=True,
        progress=False,
    )

    overall = summarise_overall(
        per_episode
    )

    return {
        **overall,
        "scenario_summary": scenario_summary.to_dict(
            orient="records"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train Double DQN with a dueling joint-action network + Representation B."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validate arguments.
    # --------------------------------------------------------------

    if args.timesteps < 1:
        raise ValueError(
            "--timesteps must be >= 1."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    if not 0.0 < args.gamma <= 1.0:
        raise ValueError(
            "--gamma must be in (0, 1]."
        )

    if args.buffer_size < args.batch_size:
        raise ValueError(
            "--buffer-size must be >= --batch-size."
        )

    if args.learning_starts < 0:
        raise ValueError(
            "--learning-starts must be >= 0."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )

    if args.train_frequency < 1:
        raise ValueError(
            "--train-frequency must be >= 1."
        )

    if args.gradient_steps < 1:
        raise ValueError(
            "--gradient-steps must be >= 1."
        )

    if args.target_update_interval < 1:
        raise ValueError(
            "--target-update-interval must be >= 1."
        )

    if not 0.0 <= args.exploration_fraction <= 1.0:
        raise ValueError(
            "--exploration-fraction must be in [0, 1]."
        )

    if not 0.0 <= args.exploration_final_eps <= 1.0:
        raise ValueError(
            "--exploration-final-eps must be in [0, 1]."
        )

    if args.max_grad_norm <= 0:
        raise ValueError(
            "--max-grad-norm must be > 0."
        )

    run_name = (
        args.run_name
        if args.run_name
        else (
            f"dueling_ddqn_"
            f"{args.timesteps // 1000}k_"
            f"lr{args.learning_rate:.0e}_"
            f"g{args.gamma}_"
            f"seed{args.seed}"
        )
    )

    save_dir = (
        MODELS_DIR
        / run_name
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Double DQN (dueling joint-action network) + Representation B ===",
        flush=True,
    )

    print(
        f"obs_dim={REPRESENTATION_B_DIM} "
        f"actions={N_ACTIONS} "
        f"timesteps={args.timesteps:,} "
        f"lr={args.learning_rate} "
        f"gamma={args.gamma} "
        f"seed={args.seed}",
        flush=True,
    )

    # --------------------------------------------------------------
    # Train.
    # --------------------------------------------------------------

    model, model_metadata = train(
        args
    )

    checkpoint_path = (
        save_dir
        / "final_model.pt"
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            **model_metadata,
            "technique": "Double DQN",
            "architecture": "Dueling 256-256 joint-action Q-network",
            "representation": "B",
        },
        checkpoint_path,
    )

    print(
        "\nTraining complete. "
        "Running official 200-episode holdout...",
        flush=True,
    )

    evaluation = evaluate_model(
        model
    )

    result = {
        "technique": "Double DQN",
        "architecture": (
            "Dueling 256-256 "
            "joint-action Q-network"
        ),
        "representation": "B",
        "observation_dim": REPRESENTATION_B_DIM,
        "n_actions": N_ACTIONS,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "train_frequency": args.train_frequency,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "max_grad_norm": args.max_grad_norm,
        "updates": model_metadata["updates"],
        "checkpoint": str(
            checkpoint_path
        ),
        "official_local_holdout": evaluation,
    }

    result_path = (
        RESULTS_DIR
        / f"{run_name}.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Double DQN (dueling joint-action network) experiment complete ===",
        flush=True,
    )

    print(
        f"Model    : {checkpoint_path}",
        flush=True,
    )

    print(
        f"Mean     : {evaluation['mean_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Std      : {evaluation['std_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Median   : {evaluation['median_cost']:,.2f}",
        flush=True,
    )

    print(
        f"Service  : {evaluation['mean_service_level']:.6f}",
        flush=True,
    )

    print(
        f"Result   : {result_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )