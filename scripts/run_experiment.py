#!/usr/bin/env python3
"""Run a configured experiment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.evaluation.rollout import (
    evaluate_policy,
    make_amq_defender_policy,
    make_bvi_defender_policy,
    random_attacker_policy,
)
from adversarial_queueing.evaluation.policy_grid import amq_policy_grid, bvi_policy_grid
from adversarial_queueing.utils.config import (
    build_amq_config,
    build_evaluation_config,
    build_policy_grid_config,
    build_service_rate_config,
    load_config,
)
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["env"]["name"] != "service_rate_control":
        raise ValueError("baseline runner currently supports only service_rate_control")

    env_config = build_service_rate_config(config)
    env = ServiceRateControlEnv(env_config)
    algorithm = config["algorithm"]["name"]
    evaluation_config = build_evaluation_config(config)
    policy_grid_config = build_policy_grid_config(config)

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "experiment"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    command = " ".join([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")

    if algorithm == "bvi":
        bvi_config = config["bvi"]
        result = run_bounded_value_iteration(
            env,
            max_queue_length=int(bvi_config["max_queue_length"]),
            tolerance=float(bvi_config["tolerance"]),
            max_iterations=int(bvi_config["max_iterations"]),
        )
        summary = {
            "algorithm": "bvi",
            "benchmark": "service_rate_control",
            "iterations": result.iterations,
            "residual": result.residual,
            "value_at_initial_state": result.values[env_config.initial_state],
            "max_queue_length": int(bvi_config["max_queue_length"]),
        }
        evaluation = evaluate_policy(
            env,
            defender_policy=make_bvi_defender_policy(result),
            attacker_policy=random_attacker_policy,
            config=evaluation_config,
        )
        write_jsonl(run_dir / "evaluation.jsonl", evaluation.rows)
        policy_rows, policy_summary = bvi_policy_grid(env, result, policy_grid_config)
        write_jsonl(run_dir / "policy_grid.jsonl", policy_rows)
        summary["evaluation"] = evaluation.summary
        summary["policy_grid"] = policy_summary
        write_json(run_dir / "summary.json", summary)
        print(f"wrote {run_dir}")
        print(
            "summary: "
            f"iterations={result.iterations} "
            f"residual={result.residual:.6g} "
            f"V0={result.values[env_config.initial_state]:.6g} "
            f"eval_avg_cost={evaluation.summary['average_cost_mean']:.6g}"
        )
        return 0

    if algorithm == "amq":
        amq_config = build_amq_config(config)
        result = LinearAMQTrainer(env, amq_config).train()
        write_jsonl(run_dir / "metrics.jsonl", result.metrics)
        trainer = LinearAMQTrainer(env, amq_config)
        trainer.weights = result.weights.copy()
        evaluation = evaluate_policy(
            env,
            defender_policy=make_amq_defender_policy(trainer),
            attacker_policy=random_attacker_policy,
            config=evaluation_config,
        )
        write_jsonl(run_dir / "evaluation.jsonl", evaluation.rows)
        policy_rows, policy_summary = amq_policy_grid(env, trainer, policy_grid_config)
        write_jsonl(run_dir / "policy_grid.jsonl", policy_rows)
        final_td_error = result.metrics[-1]["td_error"] if result.metrics else 0.0
        summary = {
            "algorithm": "amq",
            "benchmark": "service_rate_control",
            "feature_set": amq_config.feature_set,
            "total_steps": amq_config.total_steps,
            "seed": amq_config.seed,
            "final_state": result.final_state,
            "final_td_error": final_td_error,
            "weight_norm": float(np.linalg.norm(result.weights)),
            "num_logged_metrics": len(result.metrics),
            "evaluation": evaluation.summary,
            "policy_grid": policy_summary,
        }
        write_json(run_dir / "summary.json", summary)
        print(f"wrote {run_dir}")
        print(
            "summary: "
            f"steps={amq_config.total_steps} "
            f"final_td_error={final_td_error:.6g} "
            f"weight_norm={summary['weight_norm']:.6g} "
            f"eval_avg_cost={evaluation.summary['average_cost_mean']:.6g}"
        )
        return 0

    raise ValueError(f"unsupported algorithm.name: {algorithm}")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
