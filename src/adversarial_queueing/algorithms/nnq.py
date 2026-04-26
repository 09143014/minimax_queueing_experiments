"""Neural-network minimax Q-learning smoke baseline implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv


@dataclass(frozen=True)
class NNQConfig:
    hidden_size: int = 32
    learning_rate: float = 0.001
    total_steps: int = 200
    batch_size: int = 16
    replay_capacity: int = 1000
    target_update_interval: int = 50
    epsilon: float = 0.2
    seed: int = 0
    log_interval: int = 20
    state_scale: float = 10.0


@dataclass(frozen=True)
class NNQResult:
    network: "NumpyQNetwork"
    metrics: list[dict[str, Any]]
    final_state: int


class NumpyQNetwork:
    """One-hidden-layer Q network returning a Q matrix for finite actions."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator,
        state_scale: float,
    ):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_scale = state_scale
        self.w1 = rng.normal(0.0, 0.1, size=(1, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=float)
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size, dtype=float)
        self._adam_m = self._zeros_like_params()
        self._adam_v = self._zeros_like_params()
        self._adam_t = 0

    def copy(self) -> "NumpyQNetwork":
        clone = object.__new__(NumpyQNetwork)
        clone.hidden_size = self.hidden_size
        clone.output_size = self.output_size
        clone.state_scale = self.state_scale
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone._adam_m = clone._zeros_like_params()
        clone._adam_v = clone._zeros_like_params()
        clone._adam_t = 0
        return clone

    def predict(self, state: int) -> np.ndarray:
        x = self._input(state)
        hidden_pre = x @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        return (hidden @ self.w2 + self.b2).ravel()

    def q_matrix(self, state: int, num_attacker: int, num_defender: int) -> np.ndarray:
        return self.predict(state).reshape(num_attacker, num_defender)

    def train_batch(
        self,
        states: np.ndarray,
        action_indices: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = np.zeros_like(self.b2)
        losses = []

        for state, action_index, target in zip(states, action_indices, targets):
            x = self._input(int(state))
            hidden_pre = x @ self.w1 + self.b1
            hidden = np.maximum(hidden_pre, 0.0)
            output = (hidden @ self.w2 + self.b2).ravel()
            error = output[int(action_index)] - float(target)
            losses.append(error * error)

            grad_output = np.zeros(self.output_size, dtype=float)
            grad_output[int(action_index)] = 2.0 * error / len(states)
            grad_w2 += np.outer(hidden.ravel(), grad_output)
            grad_b2 += grad_output
            grad_hidden = grad_output @ self.w2.T
            grad_hidden_pre = grad_hidden * (hidden_pre.ravel() > 0.0)
            grad_w1 += np.outer(x.ravel(), grad_hidden_pre)
            grad_b1 += grad_hidden_pre

        self._adam_step(
            {
                "w1": grad_w1,
                "b1": grad_b1,
                "w2": grad_w2,
                "b2": grad_b2,
            },
            learning_rate,
        )
        return float(np.mean(losses))

    def _input(self, state: int) -> np.ndarray:
        return np.array([[float(state) / self.state_scale]], dtype=float)

    def _zeros_like_params(self) -> dict[str, np.ndarray]:
        return {
            "w1": np.zeros_like(self.w1),
            "b1": np.zeros_like(self.b1),
            "w2": np.zeros_like(self.w2),
            "b2": np.zeros_like(self.b2),
        }

    def _adam_step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self._adam_t += 1
        for name, grad in grads.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            param = getattr(self, name)
            setattr(self, name, param - learning_rate * m_hat / (np.sqrt(v_hat) + eps))


class NNQTrainer:
    """Small NNQ trainer for service-rate-control smoke experiments."""

    def __init__(self, env: ServiceRateControlEnv, config: NNQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.attacker_actions = tuple(env.attacker_actions(env.config.initial_state))
        self.defender_actions = tuple(env.defender_actions(env.config.initial_state))
        output_size = len(self.attacker_actions) * len(self.defender_actions)
        self.network = NumpyQNetwork(
            hidden_size=config.hidden_size,
            output_size=output_size,
            rng=self.rng,
            state_scale=config.state_scale,
        )
        self.target_network = self.network.copy()
        self.replay: list[tuple[int, int, int, float, int]] = []

    def train(self) -> NNQResult:
        state = int(self.env.reset(seed=self.config.seed))
        metrics: list[dict[str, Any]] = []
        last_loss = 0.0

        for step in range(1, self.config.total_steps + 1):
            attacker_action, defender_action = self._behavior_actions(state)
            next_state, cost, _info = self.env.step(attacker_action, defender_action)
            self._append_replay(state, attacker_action, defender_action, float(cost), int(next_state))

            if len(self.replay) >= self.config.batch_size:
                last_loss = self._train_one_batch()

            if step % self.config.target_update_interval == 0:
                self.target_network = self.network.copy()

            if step == 1 or step % self.config.log_interval == 0 or step == self.config.total_steps:
                metrics.append(
                    {
                        "step": step,
                        "state": state,
                        "attacker_action": attacker_action,
                        "defender_action": defender_action,
                        "next_state": int(next_state),
                        "cost": float(cost),
                        "loss": float(last_loss),
                        "q_norm": float(np.linalg.norm(self.network.predict(state))),
                        "replay_size": len(self.replay),
                    }
                )
            state = int(next_state)

        return NNQResult(network=self.network.copy(), metrics=metrics, final_state=state)

    def q_matrix(self, state: int) -> np.ndarray:
        return self.network.q_matrix(state, len(self.attacker_actions), len(self.defender_actions))

    def _behavior_actions(self, state: int) -> tuple[int, int]:
        if self.rng.random() < self.config.epsilon:
            return (
                int(self.rng.choice(self.attacker_actions)),
                int(self.rng.choice(self.defender_actions)),
            )
        game = solve_zero_sum_matrix_game(self.q_matrix(state))
        attacker = int(self.rng.choice(self.attacker_actions, p=game["attacker_strategy"]))
        defender = int(self.rng.choice(self.defender_actions, p=game["defender_strategy"]))
        return attacker, defender

    def _append_replay(
        self,
        state: int,
        attacker_action: int,
        defender_action: int,
        cost: float,
        next_state: int,
    ) -> None:
        self.replay.append((state, attacker_action, defender_action, cost, next_state))
        if len(self.replay) > self.config.replay_capacity:
            self.replay.pop(0)

    def _train_one_batch(self) -> float:
        indices = self.rng.choice(len(self.replay), size=self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        states = np.array([item[0] for item in batch], dtype=int)
        action_indices = np.array(
            [
                item[1] * len(self.defender_actions) + item[2]
                for item in batch
            ],
            dtype=int,
        )
        targets = np.array([self._target(item[3], item[4]) for item in batch], dtype=float)
        return self.network.train_batch(states, action_indices, targets, self.config.learning_rate)

    def _target(self, cost: float, next_state: int) -> float:
        next_matrix = self.target_network.q_matrix(
            next_state,
            len(self.attacker_actions),
            len(self.defender_actions),
        )
        next_value = solve_zero_sum_matrix_game(next_matrix)["value"]
        return float(cost + self.env.discount * next_value)
