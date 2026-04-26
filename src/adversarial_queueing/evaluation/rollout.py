"""Rollout evaluation for service-rate-control policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv

AttackerPolicy = Callable[[int, np.random.Generator, ServiceRateControlEnv], int]
DefenderPolicy = Callable[[int, np.random.Generator, ServiceRateControlEnv], int]


@dataclass(frozen=True)
class EvaluationConfig:
    num_episodes: int = 5
    horizon: int = 25
    seed: int = 0
    tail_threshold: int = 8
    boundary_state: int | None = None


@dataclass(frozen=True)
class RolloutResult:
    rows: list[dict[str, float | int]]
    summary: dict[str, float | int]


def evaluate_policy(
    env: ServiceRateControlEnv,
    defender_policy: DefenderPolicy,
    attacker_policy: AttackerPolicy,
    config: EvaluationConfig,
) -> RolloutResult:
    """Evaluate policies using the same environment step convention as training."""

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int]] = []

    for episode in range(config.num_episodes):
        state = int(env.reset(seed=config.seed + episode))
        discounted_cost = 0.0
        total_cost = 0.0
        tail_count = 0
        boundary_hits = 0

        for step in range(config.horizon):
            attacker_action = int(attacker_policy(state, rng, env))
            defender_action = int(defender_policy(state, rng, env))
            next_state, cost, _info = env.step(attacker_action, defender_action)
            total_cost += float(cost)
            discounted_cost += (env.discount**step) * float(cost)
            tail_count += int(next_state >= config.tail_threshold)
            if config.boundary_state is not None:
                boundary_hits += int(next_state >= config.boundary_state)
            state = int(next_state)

        rows.append(
            {
                "episode": episode,
                "seed": config.seed + episode,
                "total_cost": total_cost,
                "average_cost": total_cost / config.horizon,
                "discounted_cost": discounted_cost,
                "final_state": state,
                "tail_fraction": tail_count / config.horizon,
                "boundary_hit_fraction": boundary_hits / config.horizon,
            }
        )

    summary = _summarize_rows(rows)
    summary.update(
        {
            "num_episodes": config.num_episodes,
            "horizon": config.horizon,
            "seed": config.seed,
            "tail_threshold": config.tail_threshold,
        }
    )
    return RolloutResult(rows=rows, summary=summary)


def random_attacker_policy(
    state: int, rng: np.random.Generator, env: ServiceRateControlEnv
) -> int:
    return int(rng.choice(env.attacker_actions(state)))


def make_amq_defender_policy(trainer: LinearAMQTrainer) -> DefenderPolicy:
    def policy(state: int, rng: np.random.Generator, env: ServiceRateControlEnv) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        defender_actions = tuple(env.defender_actions(state))
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def make_nnq_defender_policy(trainer: NNQTrainer) -> DefenderPolicy:
    def policy(state: int, rng: np.random.Generator, env: ServiceRateControlEnv) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        defender_actions = tuple(env.defender_actions(state))
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def make_bvi_defender_policy(result: BVIResult) -> DefenderPolicy:
    max_state = max(result.values)

    def policy(state: int, rng: np.random.Generator, env: ServiceRateControlEnv) -> int:
        clipped_state = min(int(state), max_state)
        attacker_actions = tuple(env.attacker_actions(clipped_state))
        defender_actions = tuple(env.defender_actions(clipped_state))
        payoff = np.zeros((len(attacker_actions), len(defender_actions)), dtype=float)
        for ai, attacker_action in enumerate(attacker_actions):
            for bi, defender_action in enumerate(defender_actions):
                expected_next = 0.0
                for next_state, prob in env.transition_probabilities(
                    clipped_state, attacker_action, defender_action
                ).items():
                    expected_next += prob * result.values[min(int(next_state), max_state)]
                payoff[ai, bi] = (
                    env.cost(clipped_state, attacker_action, defender_action)
                    + env.discount * expected_next
                )
        game = solve_zero_sum_matrix_game(payoff)
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def _summarize_rows(rows: list[dict[str, float | int]]) -> dict[str, float]:
    average_costs = np.array([row["average_cost"] for row in rows], dtype=float)
    discounted_costs = np.array([row["discounted_cost"] for row in rows], dtype=float)
    final_states = np.array([row["final_state"] for row in rows], dtype=float)
    tail_fractions = np.array([row["tail_fraction"] for row in rows], dtype=float)
    boundary_fractions = np.array([row["boundary_hit_fraction"] for row in rows], dtype=float)
    return {
        "average_cost_mean": float(average_costs.mean()),
        "average_cost_std": float(average_costs.std(ddof=0)),
        "discounted_cost_mean": float(discounted_costs.mean()),
        "final_state_mean": float(final_states.mean()),
        "tail_fraction_mean": float(tail_fractions.mean()),
        "boundary_hit_fraction_mean": float(boundary_fractions.mean()),
    }
