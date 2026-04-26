"""Config loading and object construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adversarial_queueing.algorithms.amq import AMQConfig
from adversarial_queueing.envs.service_rate_control import ServiceRateControlConfig
from adversarial_queueing.evaluation.rollout import EvaluationConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def build_service_rate_config(data: dict[str, Any]) -> ServiceRateControlConfig:
    env = data["env"]
    bvi = data.get("bvi", {})
    return ServiceRateControlConfig(
        lambda_arrival=float(env["lambda_arrival"]),
        mu_levels=tuple(float(x) for x in env["mu_levels"]),
        service_costs=tuple(float(x) for x in env["service_costs"]),
        gamma=float(env.get("gamma", 0.95)),
        q_congestion=float(env.get("q_congestion", 1.0)),
        attack_cost=float(env.get("attack_cost", 0.5)),
        initial_state=int(env.get("initial_state", 0)),
        uniformization_rate=(
            None
            if env.get("uniformization_rate") is None
            else float(env["uniformization_rate"])
        ),
        robust_defender_actions=tuple(int(x) for x in env.get("robust_defender_actions", [2])),
        bvi_max_queue_length=int(bvi.get("max_queue_length", 20)),
        boundary_mode=str(bvi.get("boundary_mode", "clip")),
    )


def build_amq_config(data: dict[str, Any]) -> AMQConfig:
    amq = data.get("amq", {})
    return AMQConfig(
        feature_set=str(amq.get("feature_set", "basic_quadratic")),
        total_steps=int(amq.get("total_steps", 100)),
        eta0=float(amq.get("eta0", 0.01)),
        learning_rate_schedule=str(amq.get("learning_rate_schedule", "constant")),
        decay_power=float(amq.get("decay_power", 0.6)),
        seed=int(amq.get("seed", 0)),
        log_interval=int(amq.get("log_interval", 10)),
        weight_clip=(
            None
            if amq.get("weight_clip") is None
            else float(amq["weight_clip"])
        ),
    )


def build_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    evaluation = data.get("evaluation", {})
    return EvaluationConfig(
        num_episodes=int(evaluation.get("num_episodes", 5)),
        horizon=int(evaluation.get("horizon", 25)),
        seed=int(evaluation.get("seed", 0)),
        tail_threshold=int(evaluation.get("tail_threshold", 8)),
        boundary_state=(
            None
            if evaluation.get("boundary_state") is None
            else int(evaluation["boundary_state"])
        ),
    )
