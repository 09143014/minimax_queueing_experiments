#!/usr/bin/env python3
"""Analyze whether routing AMQ policy gaps occur in visited states."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import (
    bounded_queue_states,
    run_bounded_value_iteration,
)
from adversarial_queueing.envs.routing import RoutingEnv
from adversarial_queueing.evaluation.rollout import (
    make_amq_defender_policy,
    make_bvi_attacker_policy,
    rollout_state_visitation,
)
from adversarial_queueing.evaluation.routing_policy import (
    compare_amq_bvi_routing_policies,
)
from adversarial_queueing.utils.config import (
    build_amq_config,
    build_evaluation_config,
    build_policy_grid_config,
    build_routing_config,
    load_config,
)
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to routing AMQ YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["env"]["name"] != "routing" or config["algorithm"]["name"] != "amq":
        raise ValueError("visitation analysis requires routing AMQ config")

    env_config = build_routing_config(config)
    env = RoutingEnv(env_config)
    amq_config = build_amq_config(config)
    evaluation_config = build_evaluation_config(config)
    policy_grid_config = build_policy_grid_config(config)
    bvi_config = config["bvi"]
    max_queue_length = int(bvi_config["max_queue_length"])

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "routing_amq_visitation"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")

    result = LinearAMQTrainer(env, amq_config).train()
    trainer = LinearAMQTrainer(env, amq_config)
    trainer.weights = result.weights.copy()
    bvi_reference = run_bounded_value_iteration(
        env,
        max_queue_length=max_queue_length,
        tolerance=float(bvi_config.get("tolerance", 1e-6)),
        max_iterations=int(bvi_config.get("max_iterations", 1000)),
        states=bounded_queue_states(
            num_queues=env_config.num_queues,
            max_queue_length=max_queue_length,
        ),
    )

    comparison_rows, comparison_summary = compare_amq_bvi_routing_policies(
        env,
        trainer,
        bvi_reference,
        probability_threshold=policy_grid_config.high_probability_threshold,
    )
    visitation_rows = rollout_state_visitation(
        env,
        defender_policy=make_amq_defender_policy(trainer),
        attacker_policy=make_bvi_attacker_policy(bvi_reference),
        config=evaluation_config,
    )
    joined_rows, summary = _join_policy_and_visitation(
        comparison_rows,
        visitation_rows,
    )
    summary["policy_comparison"] = comparison_summary
    summary["evaluation"] = {
        "num_episodes": evaluation_config.num_episodes,
        "horizon": evaluation_config.horizon,
        "seed": evaluation_config.seed,
    }
    summary["training"] = {
        "feature_set": amq_config.feature_set,
        "total_steps": amq_config.total_steps,
        "seed": amq_config.seed,
    }

    write_jsonl(run_dir / "policy_visitation.jsonl", joined_rows)
    write_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")
    print(
        "summary: "
        f"visited_states={summary['num_visited_states']} "
        f"visited_over_defend_states={summary['num_visited_over_defend_states']} "
        f"weighted_gap={summary['visit_weighted_abs_gap_mean']:.6g} "
        f"unvisited_gap={summary['unvisited_abs_gap_mean']:.6g}"
    )
    return 0


def _join_policy_and_visitation(
    comparison_rows: list[dict],
    visitation_rows: list[dict],
) -> tuple[list[dict], dict]:
    visits = {
        tuple(row["state"]) if isinstance(row["state"], list) else row["state"]: row
        for row in visitation_rows
    }
    joined_rows = []
    for row in comparison_rows:
        state_key = tuple(row["state"])
        visit = visits.get(state_key, {"visit_count": 0, "visit_fraction": 0.0})
        joined = dict(row)
        joined["visit_count"] = int(visit["visit_count"])
        joined["visit_fraction"] = float(visit["visit_fraction"])
        joined["visited"] = joined["visit_count"] > 0
        joined["weighted_abs_gap"] = (
            joined["p_defend_abs_gap"] * joined["visit_fraction"]
        )
        joined_rows.append(joined)

    visited = [row for row in joined_rows if row["visited"]]
    unvisited = [row for row in joined_rows if not row["visited"]]
    visited_over = [
        row for row in visited if row["amq_over_defends"]
    ]
    visited_under = [
        row for row in visited if row["amq_under_defends"]
    ]
    return joined_rows, {
        "num_states": len(joined_rows),
        "num_visited_states": len(visited),
        "num_unvisited_states": len(unvisited),
        "num_visited_over_defend_states": len(visited_over),
        "num_visited_under_defend_states": len(visited_under),
        "visit_weighted_abs_gap_mean": float(
            sum(row["weighted_abs_gap"] for row in joined_rows)
        ),
        "visited_abs_gap_mean": _mean(visited, "p_defend_abs_gap"),
        "unvisited_abs_gap_mean": _mean(unvisited, "p_defend_abs_gap"),
        "visited_signed_gap_mean": _mean(visited, "p_defend_signed_gap"),
        "unvisited_signed_gap_mean": _mean(unvisited, "p_defend_signed_gap"),
        "top_visited_gap_states": [
            {
                "state": row["state"],
                "visit_count": row["visit_count"],
                "p_defend_amq": row["p_defend_amq"],
                "p_defend_bvi_reference": row["p_defend_bvi_reference"],
                "p_defend_signed_gap": row["p_defend_signed_gap"],
            }
            for row in sorted(
                visited,
                key=lambda item: (
                    item["visit_count"],
                    item["p_defend_abs_gap"],
                ),
                reverse=True,
            )[:10]
        ],
    }


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


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
