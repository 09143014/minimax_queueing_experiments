"""Bounded value iteration for finite truncated games."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv


@dataclass(frozen=True)
class BVIResult:
    values: dict[int, float]
    iterations: int
    residual: float


def run_bounded_value_iteration(
    env: ServiceRateControlEnv,
    max_queue_length: int,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> BVIResult:
    """Run value iteration on ``{0, ..., max_queue_length}``."""

    states = list(range(max_queue_length + 1))
    values = {state: 0.0 for state in states}

    for iteration in range(1, max_iterations + 1):
        new_values: dict[int, float] = {}
        residual = 0.0

        for state in states:
            payoff = np.zeros(
                (len(env.attacker_actions(state)), len(env.defender_actions(state))),
                dtype=float,
            )
            for ai, attacker_action in enumerate(env.attacker_actions(state)):
                for bi, defender_action in enumerate(env.defender_actions(state)):
                    expected_next = 0.0
                    for next_state, prob in env.transition_probabilities(
                        state, attacker_action, defender_action
                    ).items():
                        clipped_next = min(int(next_state), max_queue_length)
                        expected_next += prob * values[clipped_next]
                    payoff[ai, bi] = (
                        env.cost(state, attacker_action, defender_action)
                        + env.discount * expected_next
                    )

            value = solve_zero_sum_matrix_game(payoff)["value"]
            new_values[state] = value
            residual = max(residual, abs(value - values[state]))

        values = new_values
        if residual < tolerance:
            return BVIResult(values=values, iterations=iteration, residual=residual)

    return BVIResult(values=values, iterations=max_iterations, residual=residual)

