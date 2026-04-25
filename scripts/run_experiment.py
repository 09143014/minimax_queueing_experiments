#!/usr/bin/env python3
"""Run a configured experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.utils.config import build_service_rate_config, load_config
from adversarial_queueing.utils.output import create_run_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["env"]["name"] != "service_rate_control":
        raise ValueError("baseline runner currently supports only service_rate_control")
    if config["algorithm"]["name"] != "bvi":
        raise ValueError("baseline runner currently supports only algorithm.name=bvi")

    env_config = build_service_rate_config(config)
    env = ServiceRateControlEnv(env_config)
    bvi_config = config["bvi"]
    result = run_bounded_value_iteration(
        env,
        max_queue_length=int(bvi_config["max_queue_length"]),
        tolerance=float(bvi_config["tolerance"]),
        max_iterations=int(bvi_config["max_iterations"]),
    )

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "experiment"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    write_json(
        run_dir / "summary.json",
        {
            "algorithm": "bvi",
            "benchmark": "service_rate_control",
            "iterations": result.iterations,
            "residual": result.residual,
            "value_at_initial_state": result.values[env_config.initial_state],
            "max_queue_length": int(bvi_config["max_queue_length"]),
        },
    )
    print(f"wrote {run_dir}")
    print(
        "summary: "
        f"iterations={result.iterations} "
        f"residual={result.residual:.6g} "
        f"V0={result.values[env_config.initial_state]:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

