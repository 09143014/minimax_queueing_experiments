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
from adversarial_queueing.algorithms.bvi import (
    bounded_queue_states,
    run_bounded_value_iteration,
)
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.routing import RoutingEnv
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.evaluation.rollout import (
    evaluate_policy,
    make_amq_defender_policy,
    make_bvi_defender_policy,
    make_nnq_defender_policy,
    random_attacker_policy,
)
from adversarial_queueing.evaluation.routing_policy import (
    amq_routing_policy_inspection,
    bvi_routing_policy_inspection,
    compare_amq_bvi_routing_policies,
    compare_nnq_bvi_routing_policies,
    nnq_routing_policy_inspection,
    routing_amq_q_diagnostic,
    routing_nnq_q_diagnostic,
)
from adversarial_queueing.evaluation.bvi_sensitivity import run_bvi_sensitivity
from adversarial_queueing.evaluation.policy_grid import (
    amq_policy_grid,
    bvi_policy_grid,
    nnq_policy_grid,
)
from adversarial_queueing.utils.config import (
    build_amq_config,
    build_bvi_sensitivity_config,
    build_evaluation_config,
    build_nnq_config,
    build_policy_grid_config,
    build_routing_config,
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
    env_name = config["env"]["name"]
    if env_name == "service_rate_control":
        env_config = build_service_rate_config(config)
        env = ServiceRateControlEnv(env_config)
    elif env_name == "routing":
        env_config = build_routing_config(config)
        env = RoutingEnv(env_config)
    else:
        raise ValueError(f"unsupported env.name: {env_name}")

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
        max_queue_length = int(bvi_config["max_queue_length"])
        bvi_states = _bvi_states(env_name, env_config, max_queue_length)
        result = run_bounded_value_iteration(
            env,
            max_queue_length=max_queue_length,
            tolerance=float(bvi_config["tolerance"]),
            max_iterations=int(bvi_config["max_iterations"]),
            states=bvi_states,
        )
        summary = {
            "algorithm": "bvi",
            "benchmark": env_name,
            "iterations": result.iterations,
            "residual": result.residual,
            "value_at_initial_state": result.values[_initial_state(env_name, env_config)],
            "max_queue_length": max_queue_length,
            "num_states": len(result.values),
        }
        if env_name == "routing":
            evaluation = evaluate_policy(
                env,
                defender_policy=make_bvi_defender_policy(result),
                attacker_policy=random_attacker_policy,
                config=evaluation_config,
            )
            write_jsonl(run_dir / "evaluation.jsonl", evaluation.rows)
            policy_rows, policy_summary = bvi_routing_policy_inspection(
                env,
                result,
                probability_threshold=policy_grid_config.high_probability_threshold,
            )
            write_jsonl(run_dir / "policy_inspection.jsonl", policy_rows)
            summary["evaluation"] = evaluation.summary
            summary["policy_inspection"] = policy_summary
            write_json(run_dir / "summary.json", summary)
            print(f"wrote {run_dir}")
            print(
                "summary: "
                f"states={len(result.values)} "
                f"iterations={result.iterations} "
                f"residual={result.residual:.6g} "
                f"V0={summary['value_at_initial_state']:.6g} "
                f"eval_avg_cost={evaluation.summary['average_cost_mean']:.6g} "
                f"defend_states={policy_summary['num_states_p_defend_at_least_threshold']}"
            )
            return 0

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
        if env_name == "service_rate_control":
            policy_rows, policy_summary = amq_policy_grid(env, trainer, policy_grid_config)
            write_jsonl(run_dir / "policy_grid.jsonl", policy_rows)
            policy_summary_key = "policy_grid"
            policy_summary_value = policy_summary
        elif env_name == "routing":
            bvi_config = config.get("bvi", {})
            max_queue_length = int(
                bvi_config.get(
                    "max_queue_length",
                    config.get("policy_grid", {}).get("max_state", 3),
                )
            )
            policy_rows, policy_summary = amq_routing_policy_inspection(
                env,
                trainer,
                max_queue_length=max_queue_length,
                probability_threshold=policy_grid_config.high_probability_threshold,
            )
            write_jsonl(run_dir / "policy_inspection.jsonl", policy_rows)
            bvi_reference = run_bounded_value_iteration(
                env,
                max_queue_length=max_queue_length,
                tolerance=float(bvi_config.get("tolerance", 1e-6)),
                max_iterations=int(bvi_config.get("max_iterations", 1000)),
                states=_bvi_states(env_name, env_config, max_queue_length),
            )
            comparison_rows, comparison_summary = compare_amq_bvi_routing_policies(
                env,
                trainer,
                bvi_reference,
                probability_threshold=policy_grid_config.high_probability_threshold,
            )
            write_jsonl(run_dir / "policy_comparison.jsonl", comparison_rows)
            q_rows, q_summary = routing_amq_q_diagnostic(env, trainer, bvi_reference)
            write_jsonl(run_dir / "q_diagnostic.jsonl", q_rows)
            policy_summary_key = "policy_inspection"
            policy_summary_value = policy_summary
            extra_summary = {
                "policy_comparison": comparison_summary,
                "q_diagnostic": q_summary,
                "bvi_reference": {
                    "iterations": bvi_reference.iterations,
                    "residual": bvi_reference.residual,
                    "max_queue_length": max_queue_length,
                    "num_states": len(bvi_reference.values),
                    "role": "bounded_reference_for_evaluation_only",
                },
            }
        else:
            raise ValueError(f"AMQ runner does not support env.name: {env_name}")
        final_td_error = result.metrics[-1]["td_error"] if result.metrics else 0.0
        summary = {
            "algorithm": "amq",
            "benchmark": env_name,
            "feature_set": amq_config.feature_set,
            "total_steps": amq_config.total_steps,
            "seed": amq_config.seed,
            "final_state": _json_state(result.final_state),
            "final_td_error": final_td_error,
            "weight_norm": float(np.linalg.norm(result.weights)),
            "num_logged_metrics": len(result.metrics),
            "evaluation": evaluation.summary,
            policy_summary_key: policy_summary_value,
        }
        if env_name == "routing":
            summary.update(extra_summary)
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

    if algorithm == "nnq":
        nnq_config = build_nnq_config(config)
        trainer = NNQTrainer(env, nnq_config)
        result = trainer.train()
        trainer.network = result.network.copy()
        trainer.target_network = result.network.copy()
        write_jsonl(run_dir / "metrics.jsonl", result.metrics)
        evaluation = evaluate_policy(
            env,
            defender_policy=make_nnq_defender_policy(trainer),
            attacker_policy=random_attacker_policy,
            config=evaluation_config,
        )
        write_jsonl(run_dir / "evaluation.jsonl", evaluation.rows)
        if env_name == "service_rate_control":
            policy_rows, policy_summary = nnq_policy_grid(env, trainer, policy_grid_config)
            write_jsonl(run_dir / "policy_grid.jsonl", policy_rows)
            policy_summary_key = "policy_grid"
        elif env_name == "routing":
            max_queue_length = int(config.get("bvi", {}).get("max_queue_length", 3))
            policy_rows, policy_summary = nnq_routing_policy_inspection(
                env,
                trainer,
                max_queue_length=max_queue_length,
                probability_threshold=policy_grid_config.high_probability_threshold,
            )
            write_jsonl(run_dir / "policy_inspection.jsonl", policy_rows)
            policy_summary_key = "policy_inspection"
            bvi_result = run_bounded_value_iteration(
                env,
                max_queue_length=max_queue_length,
                tolerance=float(config.get("bvi", {}).get("tolerance", 1e-6)),
                max_iterations=int(config.get("bvi", {}).get("max_iterations", 1000)),
                states=_bvi_states(env_name, env_config, max_queue_length),
            )
            comparison_rows, comparison_summary = compare_nnq_bvi_routing_policies(
                env,
                trainer,
                bvi_result,
                probability_threshold=policy_grid_config.high_probability_threshold,
            )
            write_jsonl(run_dir / "policy_comparison.jsonl", comparison_rows)
            q_rows, q_summary = routing_nnq_q_diagnostic(env, trainer, bvi_result)
            write_jsonl(run_dir / "q_diagnostic.jsonl", q_rows)
        else:
            raise ValueError(f"NNQ runner does not support env.name: {env_name}")
        final_loss = result.metrics[-1]["loss"] if result.metrics else 0.0
        summary = {
            "algorithm": "nnq",
            "benchmark": env_name,
            "hidden_size": nnq_config.hidden_size,
            "state_feature_set": nnq_config.state_feature_set,
            "forced_defender_action_probability": (
                nnq_config.forced_defender_action_probability
            ),
            "forced_defender_action": nnq_config.forced_defender_action,
            "total_steps": nnq_config.total_steps,
            "seed": nnq_config.seed,
            "final_state": _json_state(result.final_state),
            "final_loss": final_loss,
            "num_logged_metrics": len(result.metrics),
            "evaluation": evaluation.summary,
            policy_summary_key: policy_summary,
            "implementation": "numpy_mlp_smoke",
        }
        if env_name == "routing":
            summary["policy_comparison"] = comparison_summary
            summary["q_diagnostic"] = q_summary
            summary["bvi_reference"] = {
                "role": "bounded_reference_for_evaluation_only",
                "max_queue_length": max_queue_length,
                "iterations": bvi_result.iterations,
                "residual": bvi_result.residual,
            }
        write_json(run_dir / "summary.json", summary)
        print(f"wrote {run_dir}")
        print(
            "summary: "
            f"steps={nnq_config.total_steps} "
            f"final_loss={final_loss:.6g} "
            f"eval_avg_cost={evaluation.summary['average_cost_mean']:.6g}"
        )
        return 0

    if algorithm == "bvi_sensitivity":
        if env_name != "service_rate_control":
            raise ValueError("BVI sensitivity currently supports only service_rate_control")
        sensitivity_config = build_bvi_sensitivity_config(config)
        rows, sensitivity_summary = run_bvi_sensitivity(
            env_config,
            sensitivity_config,
            evaluation_config,
            policy_grid_config,
        )
        write_jsonl(run_dir / "sensitivity.jsonl", rows)
        summary = {
            "algorithm": "bvi_sensitivity",
            "benchmark": "service_rate_control",
            "sensitivity": sensitivity_summary,
        }
        write_json(run_dir / "summary.json", summary)
        print(f"wrote {run_dir}")
        print(
            "summary: "
            f"bounds={sensitivity_summary['bounds']} "
            f"value_range={sensitivity_summary['value_range_at_eval_state']:.6g} "
            f"threshold_stable={sensitivity_summary['policy_high_threshold_stable']}"
        )
        return 0

    raise ValueError(f"unsupported algorithm.name: {algorithm}")


def _bvi_states(env_name: str, env_config, max_queue_length: int):
    if env_name == "service_rate_control":
        return bounded_queue_states(num_queues=1, max_queue_length=max_queue_length)
    if env_name == "routing":
        return bounded_queue_states(
            num_queues=env_config.num_queues,
            max_queue_length=max_queue_length,
        )
    raise ValueError(f"unsupported env.name for BVI: {env_name}")


def _initial_state(env_name: str, env_config):
    if env_name == "service_rate_control":
        return env_config.initial_state
    if env_name == "routing":
        return env_config.initial_state_value
    raise ValueError(f"unsupported env.name for initial state: {env_name}")


def _json_state(state):
    if isinstance(state, tuple):
        return [int(value) for value in state]
    return int(state)


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
