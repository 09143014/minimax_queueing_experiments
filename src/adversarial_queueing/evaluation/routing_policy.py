"""Policy inspection helpers for the routing benchmark."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.routing import RoutingEnv, State


def bvi_routing_policy_inspection(
    env: RoutingEnv,
    result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect BVI defender strategies across the bounded routing state space."""

    rows: list[dict[str, Any]] = []
    for state in sorted(result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing policy inspection requires tuple states")
        game = _bvi_game_at_state(env, result, state)
        defender_strategy = game["defender_strategy"]
        rows.append(
            {
                "method": "bvi",
                "state": list(state),
                "total_queue": sum(state),
                "imbalance": max(state) - min(state),
                "nominal_targets": list(
                    env.routed_arrival_targets(state, attacker_action=0, defender_action=0)
                ),
                "attacked_targets": list(
                    env.routed_arrival_targets(state, attacker_action=1, defender_action=0)
                ),
                "p_no_defend": float(defender_strategy[0]),
                "p_defend": float(defender_strategy[1]),
                "value": float(game["value"]),
            }
        )
    return rows, _summary(rows, probability_threshold)


def _bvi_game_at_state(env: RoutingEnv, result: BVIResult, state: State) -> dict[str, Any]:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(max(value) for value in result.values if isinstance(value, tuple))

    attacker_actions = tuple(env.attacker_actions(state))
    defender_actions = tuple(env.defender_actions(state))
    payoff = np.zeros((len(attacker_actions), len(defender_actions)), dtype=float)
    for ai, attacker_action in enumerate(attacker_actions):
        for bi, defender_action in enumerate(defender_actions):
            expected_next = 0.0
            for next_state, prob in env.transition_probabilities(
                state, attacker_action, defender_action
            ).items():
                expected_next += prob * result.values[
                    _bounded_state(next_state, max_queue_length)
                ]
            payoff[ai, bi] = (
                env.cost(state, attacker_action, defender_action)
                + env.discount * expected_next
            )
    return solve_zero_sum_matrix_game(payoff)


def _summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
) -> dict[str, Any]:
    defend_probs = np.array([row["p_defend"] for row in rows], dtype=float)
    defend_rows = [row for row in rows if row["p_defend"] >= probability_threshold]
    return {
        "num_policy_states": len(rows),
        "defend_probability_mean": float(defend_probs.mean()),
        "defend_probability_max": float(defend_probs.max()),
        "defend_probability_threshold": probability_threshold,
        "num_states_p_defend_at_least_threshold": len(defend_rows),
        "first_state_p_defend_at_least_threshold": (
            None if not defend_rows else defend_rows[0]["state"]
        ),
    }


def _bounded_state(state: Hashable, max_queue_length: int) -> Hashable:
    if isinstance(state, tuple):
        return tuple(min(int(value), max_queue_length) for value in state)
    return min(int(state), max_queue_length)


def _state_sort_key(state: Hashable) -> tuple[int, tuple[int, ...]]:
    if not isinstance(state, tuple):
        return (int(state), (int(state),))
    return (sum(state), tuple(int(value) for value in state))
