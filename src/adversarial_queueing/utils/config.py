"""Config loading and object construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adversarial_queueing.envs.service_rate_control import ServiceRateControlConfig


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

