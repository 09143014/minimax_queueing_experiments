"""Approximate minimax Q-learning with linear function approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
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


@dataclass(frozen=True)
class AMQResult:
    weights: np.ndarray
    metrics: list[dict[str, Any]]
    final_state: int


class LinearAMQTrainer:
    """Small AMQ trainer used for service-rate-control smoke experiments."""

    def __init__(self, env: ServiceRateControlEnv, config: AMQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.attacker_actions = tuple(env.attacker_actions(env.config.initial_state))
        self.defender_actions = tuple(env.defender_actions(env.config.initial_state))
        dim = service_rate_feature_dim(
            feature_set=config.feature_set,
            num_attacker_actions=len(self.attacker_actions),
            num_defender_actions=len(self.defender_actions),
        )
        self.weights = np.zeros(dim, dtype=float)

    def train(self) -> AMQResult:
        state = self.env.reset(seed=self.config.seed)
        metrics: list[dict[str, Any]] = []

        for step in range(1, self.config.total_steps + 1):
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
                        "state": int(state),
                        "attacker_action": attacker_action,
                        "defender_action": defender_action,
                        "next_state": int(next_state),
                        "cost": float(cost),
                        "td_error": td_error,
                        "weight_norm": float(np.linalg.norm(self.weights)),
                        "feature_norm": float(np.linalg.norm(phi)),
                        "minimax_value_next": float(next_value),
                    }
                )
            state = int(next_state)

        return AMQResult(weights=self.weights.copy(), metrics=metrics, final_state=int(state))

    def q_value(self, state: int, attacker_action: int, defender_action: int) -> float:
        return float(self._features(state, attacker_action, defender_action) @ self.weights)

    def q_matrix(self, state: int) -> np.ndarray:
        matrix = np.zeros((len(self.attacker_actions), len(self.defender_actions)), dtype=float)
        for ai, attacker_action in enumerate(self.attacker_actions):
            for bi, defender_action in enumerate(self.defender_actions):
                matrix[ai, bi] = self.q_value(state, attacker_action, defender_action)
        return matrix

    def value(self, state: int) -> float:
        return float(solve_zero_sum_matrix_game(self.q_matrix(state))["value"])

    def _features(self, state: int, attacker_action: int, defender_action: int) -> np.ndarray:
        return service_rate_features(
            state=state,
            attacker_action=attacker_action,
            defender_action=defender_action,
            feature_set=self.config.feature_set,
            num_attacker_actions=len(self.attacker_actions),
            num_defender_actions=len(self.defender_actions),
        )

    def _learning_rate(self, step: int) -> float:
        if self.config.learning_rate_schedule == "constant":
            return float(self.config.eta0)
        if self.config.learning_rate_schedule == "robbins_monro":
            return float(self.config.eta0 / (step**self.config.decay_power))
        raise ValueError(f"unknown learning_rate_schedule: {self.config.learning_rate_schedule}")

