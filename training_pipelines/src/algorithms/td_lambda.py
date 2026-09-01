
"""True Online SARSA(lambda) with linear function approximation and a reduced
action catalogue (Phase 5, section 8).

Traces reset at episode boundaries (doc 8.3), so training is organized by
full episodes rather than a flat transition counter, unlike Neural SARSA.
Uses the "true online" update (van Seijen et al. 2016 / Sutton & Barto 2nd
ed. section 12.7), which corrects the double-counting that plain
accumulating-trace SARSA(lambda) would introduce under linear function
approximation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common.action_catalogue import ACTION_CATALOGUE, CATALOGUE_SIZE, catalogue_action_to_env_indices
from .common.discretizer import FEATURE_DIM, flatten_observation_bands


@dataclass
class TDLambdaConfig:
    gamma: float = 0.99
    lambda_: float = 0.85
    alpha: float = 0.01
    epsilon_start: float = 0.4
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 10_000
    seed: int = 20260825
    total_episodes: int = 2_000
    eval_every: int = 200
    eval_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(900, 910)))


class LinearSarsaLambda:
    """Q(s, a) = W[a] . phi(s), one weight row per catalogue action."""

    def __init__(self, n_actions: int = CATALOGUE_SIZE, n_features: int = FEATURE_DIM) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.W = np.zeros((n_actions, n_features), dtype=np.float64)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.W @ features

    def save(self, path: str | Path) -> None:
        np.savez(str(path), W=self.W)

    @classmethod
    def load(cls, path: str | Path) -> "LinearSarsaLambda":
        data = np.load(str(path))
        W = data["W"]
        model = cls(n_actions=W.shape[0], n_features=W.shape[1])
        model.W = W
        return model


def _epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def _epsilon_schedule(episode: int, cfg: TDLambdaConfig) -> float:
    frac = min(1.0, episode / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def _evaluate_greedy(
    model: LinearSarsaLambda, eval_env_fn: Callable[[], Any], seeds
) -> tuple[float, float]:
    env = eval_env_fn()
    costs, services = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        ep_cost = 0.0
        ep_demand = ep_fulfilled = 0
        while not done:
            action_index = int(np.argmax(model.q_values(flatten_observation_bands(obs))))
            obs, _reward, terminated, truncated, info = env.step(
                catalogue_action_to_env_indices(action_index)
            )
            ep_cost += float(info["costs"]["daily_total"])
            ep_demand += int(np.sum(info["demand"]))
            ep_fulfilled += int(np.sum(info["fulfilled_demand"]))
            done = bool(terminated or truncated)
        costs.append(ep_cost)
        services.append(ep_fulfilled / max(ep_demand, 1))
    return float(np.mean(costs)), float(np.mean(services))


def train_td_lambda(
    env_fn: Callable[[], Any],
    eval_env_fn: Callable[[], Any],
    save_dir: str | Path,
    cfg: TDLambdaConfig = TDLambdaConfig(),
    verbose: bool = True,
) -> dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    model = LinearSarsaLambda(CATALOGUE_SIZE, FEATURE_DIM)
    env = env_fn()

    history: list[dict[str, Any]] = []
    best_eval_cost = float("inf")
    started_at = time.perf_counter()

    for episode in range(1, cfg.total_episodes + 1):
        epsilon = _epsilon_schedule(episode, cfg)
        z = np.zeros_like(model.W)

        obs, _ = env.reset(seed=cfg.seed + episode)
        features = flatten_observation_bands(obs)
        action = _epsilon_greedy_action(model.q_values(features), epsilon, rng)
        q_old = 0.0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _info = env.step(
                catalogue_action_to_env_indices(action)
            )
            done = bool(terminated or truncated)
            next_features = flatten_observation_bands(next_obs)
            next_action = _epsilon_greedy_action(model.q_values(next_features), epsilon, rng)

            q_sa = float(model.W[action] @ features)
            q_next = 0.0 if done else float(model.W[next_action] @ next_features)
            delta = reward + cfg.gamma * q_next - q_sa

            z *= cfg.gamma * cfg.lambda_
            trace_dot = float(z[action] @ features)
            z[action] += (1.0 - cfg.alpha * cfg.gamma * cfg.lambda_ * trace_dot) * features

            model.W += cfg.alpha * (delta + q_sa - q_old) * z
            model.W[action] -= cfg.alpha * (q_sa - q_old) * features

            q_old = q_next
            features = next_features
            action = next_action

        if episode % cfg.eval_every == 0 or episode == cfg.total_episodes:
            eval_mean_cost, eval_mean_service = _evaluate_greedy(model, eval_env_fn, cfg.eval_seeds)
            record = {
                "episode": episode,
                "epsilon": epsilon,
                "eval_mean_cost": eval_mean_cost,
                "eval_mean_service": eval_mean_service,
            }
            if eval_mean_cost < best_eval_cost:
                best_eval_cost = eval_mean_cost
                model.save(save_dir / "policy_weights.npz")
                record["saved_best"] = True
            history.append(record)
            if verbose:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[episode {episode:>6}/{cfg.total_episodes}] eps={epsilon:.3f} "
                    f"eval_cost={eval_mean_cost:,.1f} eval_service={eval_mean_service:.3f} "
                    f"elapsed={elapsed:.1f}s" + (" *new best*" if record.get("saved_best") else ""),
                    flush=True,
                )

    model.save(save_dir / "policy_weights_final.npz")
    return {"history": history, "best_eval_cost": best_eval_cost}