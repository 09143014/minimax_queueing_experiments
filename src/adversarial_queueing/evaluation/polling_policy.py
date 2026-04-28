"""Policy inspection helpers for the polling benchmark."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.polling import PollingEnv, State


def bvi_polling_policy_inspection(
    env: PollingEnv,
    result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("polling policy inspection requires tuple states")
        game = _bvi_game_at_state(env, result, state)
        rows.append(_policy_row("bvi", env, state, game))
    return rows, _summary(rows, probability_threshold)


def amq_polling_policy_inspection(
    env: PollingEnv,
    trainer: LinearAMQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(
        _polling_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    ):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("amq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def nnq_polling_policy_inspection(
    env: PollingEnv,
    trainer: NNQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(
        _polling_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    ):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("nnq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def _bvi_game_at_state(env: PollingEnv, result: BVIResult, state: State) -> dict[str, Any]:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(
            max(value[:-1]) for value in result.values if isinstance(value, tuple)
        )

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
                    _bounded_polling_state(next_state, max_queue_length)
                ]
            payoff[ai, bi] = (
                env.cost(state, attacker_action, defender_action)
                + env.discount * expected_next
            )
    return solve_zero_sum_matrix_game(payoff)


def _policy_row(
    method: str,
    env: PollingEnv,
    state: State,
    game: dict[str, Any],
) -> dict[str, Any]:
    defender_strategy = game["defender_strategy"]
    if defender_strategy.shape[0] != 2:
        raise ValueError("polling policy inspection expects two defender actions")
    queues = tuple(int(value) for value in state[:-1])
    position = int(state[-1])
    gap = max(queues) - min(queues)
    return {
        "method": method,
        "state": list(state),
        "queues": list(queues),
        "position": position,
        "total_queue": sum(queues),
        "queue_gap": gap,
        "nominal_targets": list(
            env.polling_targets(state, attacker_action=0, defender_action=0)
        ),
        "attacked_targets": list(
            env.polling_targets(state, attacker_action=1, defender_action=0)
        ),
        "p_no_defend": float(defender_strategy[0]),
        "p_defend": float(defender_strategy[1]),
        "value": float(game["value"]),
    }


def _summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
) -> dict[str, Any]:
    defend_probs = np.array([row["p_defend"] for row in rows], dtype=float)
    defend_rows = [row for row in rows if row["p_defend"] >= probability_threshold]
    gap_rows = [row for row in rows if int(row["queue_gap"]) > 0]
    gap_defend_rows = [
        row
        for row in gap_rows
        if row["p_defend"] >= probability_threshold
    ]
    return {
        "num_policy_states": len(rows),
        "defend_probability_mean": float(defend_probs.mean()),
        "defend_probability_max": float(defend_probs.max()),
        "defend_probability_threshold": probability_threshold,
        "num_states_p_defend_at_least_threshold": len(defend_rows),
        "first_state_p_defend_at_least_threshold": (
            None if not defend_rows else defend_rows[0]["state"]
        ),
        "num_gap_states": len(gap_rows),
        "num_gap_states_p_defend_at_least_threshold": len(gap_defend_rows),
        "by_queue_gap": _group_summary(rows, "queue_gap"),
        "by_total_queue": _group_summary(rows, "total_queue"),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = sorted({int(row[key]) for row in rows})
    summaries = []
    for group in groups:
        group_rows = [row for row in rows if int(row[key]) == group]
        defend_probs = np.array([row["p_defend"] for row in group_rows], dtype=float)
        summaries.append(
            {
                key: group,
                "num_states": len(group_rows),
                "p_defend_mean": float(defend_probs.mean()),
                "p_defend_max": float(defend_probs.max()),
            }
        )
    return summaries


def _polling_states(num_queues: int, max_queue_length: int) -> list[State]:
    queue_states: list[tuple[int, ...]] = [()]
    for _ in range(num_queues):
        queue_states = [
            (*prefix, value)
            for prefix in queue_states
            for value in range(max_queue_length + 1)
        ]
    return [
        (*queues, position)
        for queues in queue_states
        for position in range(num_queues)
    ]


def _bounded_polling_state(state: Hashable, max_queue_length: int) -> State:
    if not isinstance(state, tuple):
        raise ValueError("polling bounded state requires tuple state")
    queues = tuple(min(int(value), max_queue_length) for value in state[:-1])
    return (*queues, int(state[-1]))


def _state_sort_key(state: Hashable) -> tuple[int, tuple[int, ...]]:
    if not isinstance(state, tuple):
        return (int(state), (int(state),))
    queues = tuple(int(value) for value in state[:-1])
    return (sum(queues), (*queues, int(state[-1])))
