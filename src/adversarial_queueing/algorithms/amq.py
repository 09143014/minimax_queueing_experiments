"""Approximate minimax Q-learning with linear function approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.base import BaseAdversarialQueueEnv
from adversarial_queueing.envs.routing import RoutingEnv
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.features.routing_features import routing_feature_dim, routing_features
from adversarial_queueing.features.service_rate_features import (
    service_rate_feature_dim,
    service_rate_features,
)


@dataclass(frozen=True)
class AMQConfig:
    feature_set: str = "basic_quadratic"
    total_steps: int = 100
    eta0: float = 0.01
    learning_rate_schedule: str = "constant"
    decay_power: float = 0.6
    seed: int = 0
    log_interval: int = 10
    weight_clip: float | None = None
    exploring_starts_probability: float = 0.0
    exploring_starts_max_queue_length: int | None = None
    fitted_calibration_passes: int = 0
    fitted_calibration_max_queue_length: int | None = None
    fitted_calibration_eta: float | None = None


@dataclass(frozen=True)
class AMQResult:
    weights: np.ndarray
    metrics: list[dict[str, Any]]
    final_state: Hashable


class LinearAMQTrainer:
    """Small AMQ trainer used for benchmark smoke experiments."""

    def __init__(self, env: BaseAdversarialQueueEnv, config: AMQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        initial_state = self._initial_state()
        self.attacker_actions = tuple(env.attacker_actions(initial_state))
        self.defender_actions = tuple(env.defender_actions(initial_state))
        dim = self._feature_dim()
        self.weights = np.zeros(dim, dtype=float)

    def train(self) -> AMQResult:
        state = self.env.reset(seed=self.config.seed)
        metrics: list[dict[str, Any]] = []

        for step in range(1, self.config.total_steps + 1):
            state = self._maybe_exploring_start(state)
            attacker_action = int(self.rng.choice(self.attacker_actions))
            defender_action = int(self.rng.choice(self.defender_actions))
            next_state, cost, _info = self.env.step(attacker_action, defender_action)

            phi = self._features(state, attacker_action, defender_action)
            current_q = float(phi @ self.weights)
            next_value = self.value(next_state)
            td_error = float(cost + self.env.discount * next_value - current_q)
            eta = self._learning_rate(step)
            self.weights = self.weights + eta * phi * td_error
            if self.config.weight_clip is not None:
                clip = float(self.config.weight_clip)
                self.weights = np.clip(self.weights, -clip, clip)

            if step == 1 or step % self.config.log_interval == 0 or step == self.config.total_steps:
                metrics.append(
                    {
                        "step": step,
                        "state": _json_state(state),
                        "attacker_action": attacker_action,
                        "defender_action": defender_action,
                        "next_state": _json_state(next_state),
                        "cost": float(cost),
                        "td_error": td_error,
                        "weight_norm": float(np.linalg.norm(self.weights)),
                        "feature_norm": float(np.linalg.norm(phi)),
                        "minimax_value_next": float(next_value),
                    }
                )
            state = next_state

        self._fitted_calibration(metrics)

        return AMQResult(weights=self.weights.copy(), metrics=metrics, final_state=state)

    def q_value(self, state: Hashable, attacker_action: int, defender_action: int) -> float:
        return float(self._features(state, attacker_action, defender_action) @ self.weights)

    def q_matrix(self, state: Hashable) -> np.ndarray:
        matrix = np.zeros((len(self.attacker_actions), len(self.defender_actions)), dtype=float)
        for ai, attacker_action in enumerate(self.attacker_actions):
            for bi, defender_action in enumerate(self.defender_actions):
                matrix[ai, bi] = self.q_value(state, attacker_action, defender_action)
        return matrix

    def value(self, state: Hashable) -> float:
        return float(solve_zero_sum_matrix_game(self.q_matrix(state))["value"])

    def _features(
        self, state: Hashable, attacker_action: int, defender_action: int
    ) -> np.ndarray:
        if isinstance(self.env, ServiceRateControlEnv):
            return service_rate_features(
                state=int(state),
                attacker_action=attacker_action,
                defender_action=defender_action,
                feature_set=self.config.feature_set,
                num_attacker_actions=len(self.attacker_actions),
                num_defender_actions=len(self.defender_actions),
            )
        if isinstance(self.env, RoutingEnv):
            if not isinstance(state, tuple):
                raise ValueError("routing AMQ requires tuple states")
            return routing_features(
                state=state,
                attacker_action=attacker_action,
                defender_action=defender_action,
                feature_set=self.config.feature_set,
            )
        raise ValueError(f"unsupported AMQ environment: {type(self.env).__name__}")

    def _feature_dim(self) -> int:
        if isinstance(self.env, ServiceRateControlEnv):
            return service_rate_feature_dim(
                feature_set=self.config.feature_set,
                num_attacker_actions=len(self.attacker_actions),
                num_defender_actions=len(self.defender_actions),
            )
        if isinstance(self.env, RoutingEnv):
            return routing_feature_dim(
                num_queues=self.env.config.num_queues,
                feature_set=self.config.feature_set,
            )
        raise ValueError(f"unsupported AMQ environment: {type(self.env).__name__}")

    def _initial_state(self) -> Hashable:
        if isinstance(self.env, ServiceRateControlEnv):
            return self.env.config.initial_state
        if isinstance(self.env, RoutingEnv):
            return self.env.config.initial_state_value
        return self.env.reset(seed=self.config.seed)

    def _learning_rate(self, step: int) -> float:
        if self.config.learning_rate_schedule == "constant":
            return float(self.config.eta0)
        if self.config.learning_rate_schedule == "robbins_monro":
            return float(self.config.eta0 / (step**self.config.decay_power))
        raise ValueError(f"unknown learning_rate_schedule: {self.config.learning_rate_schedule}")

    def _maybe_exploring_start(self, state: Hashable) -> Hashable:
        probability = self.config.exploring_starts_probability
        if probability <= 0.0:
            return state
        if probability > 1.0:
            raise ValueError("exploring_starts_probability must be in [0, 1]")
        if self.rng.random() >= probability:
            return state
        if self.config.exploring_starts_max_queue_length is None:
            raise ValueError("exploring_starts_max_queue_length is required when exploring starts are enabled")
        if isinstance(self.env, RoutingEnv):
            bound = int(self.config.exploring_starts_max_queue_length)
            if bound < 0:
                raise ValueError("exploring_starts_max_queue_length must be nonnegative")
            sampled = tuple(
                int(self.rng.integers(0, bound + 1))
                for _ in range(self.env.config.num_queues)
            )
            return self.env.set_state(sampled)
        raise ValueError("exploring starts are currently implemented only for routing AMQ")

    def _fitted_calibration(self, metrics: list[dict[str, Any]]) -> None:
        passes = int(self.config.fitted_calibration_passes)
        if passes <= 0:
            return
        if not isinstance(self.env, RoutingEnv):
            raise ValueError("fitted calibration is currently implemented only for routing AMQ")
        if self.config.fitted_calibration_max_queue_length is None:
            raise ValueError("fitted_calibration_max_queue_length is required")
        eta = (
            float(self.config.fitted_calibration_eta)
            if self.config.fitted_calibration_eta is not None
            else float(self.config.eta0)
        )
        if eta <= 0.0:
            raise ValueError("fitted_calibration_eta must be positive")

        bound = int(self.config.fitted_calibration_max_queue_length)
        states = _bounded_routing_states(self.env.config.num_queues, bound)
        updates = 0
        abs_td_error_sum = 0.0
        last_td_error = 0.0
        for _calibration_pass in range(passes):
            for state in states:
                for attacker_action in self.attacker_actions:
                    for defender_action in self.defender_actions:
                        phi = self._features(state, attacker_action, defender_action)
                        current_q = float(phi @ self.weights)
                        target = self._expected_bellman_target(
                            state,
                            attacker_action,
                            defender_action,
                        )
                        td_error = float(target - current_q)
                        self.weights = self.weights + eta * phi * td_error
                        if self.config.weight_clip is not None:
                            clip = float(self.config.weight_clip)
                            self.weights = np.clip(self.weights, -clip, clip)
                        updates += 1
                        abs_td_error_sum += abs(td_error)
                        last_td_error = td_error
        metrics.append(
            {
                "step": self.config.total_steps,
                "phase": "fitted_calibration",
                "passes": passes,
                "num_updates": updates,
                "td_error": last_td_error,
                "mean_abs_td_error": abs_td_error_sum / updates if updates else 0.0,
                "weight_norm": float(np.linalg.norm(self.weights)),
            }
        )

    def _expected_bellman_target(
        self,
        state: Hashable,
        attacker_action: int,
        defender_action: int,
    ) -> float:
        expected_next = 0.0
        for next_state, probability in self.env.transition_probabilities(
            state,
            attacker_action,
            defender_action,
        ).items():
            expected_next += probability * self.value(next_state)
        return float(
            self.env.cost(state, attacker_action, defender_action)
            + self.env.discount * expected_next
        )


def _json_state(state: Hashable) -> int | list[int]:
    if isinstance(state, tuple):
        return [int(value) for value in state]
    return int(state)


def _bounded_routing_states(num_queues: int, max_queue_length: int) -> tuple[tuple[int, ...], ...]:
    if max_queue_length < 0:
        raise ValueError("max_queue_length must be nonnegative")
    states = [()]
    for _ in range(num_queues):
        states = [
            (*prefix, value)
            for prefix in states
            for value in range(max_queue_length + 1)
        ]
    return tuple(states)
