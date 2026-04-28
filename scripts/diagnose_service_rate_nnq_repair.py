#!/usr/bin/env python3
"""Evaluate simple low-state NNQ policy repairs for service-rate control."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Hashable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQConfig
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.base import BaseAdversarialQueueEnv
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.evaluation.rollout import (
    DefenderPolicy,
    EvaluationConfig,
    evaluate_policy,
    make_nnq_defender_policy,
    random_attacker_policy,
)
from adversarial_queueing.utils.config import (
    build_evaluation_config,
    build_nnq_config,
    build_policy_grid_config,
    build_service_rate_config,
    load_config,
)
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--json-output", help="Optional JSON report path.")
    parser.add_argument("--markdown-output", help="Optional Markdown report path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["env"]["name"] != "service_rate_control":
        raise ValueError("service-rate NNQ repair diagnostic requires service_rate_control")
    if config["algorithm"]["name"] != "nnq":
        raise ValueError("service-rate NNQ repair diagnostic requires algorithm.name=nnq")

    env_config = build_service_rate_config(config)
    base_nnq_config = build_nnq_config(config)
    evaluation_config = build_evaluation_config(config)
    policy_grid_config = build_policy_grid_config(config)
    repair_config = config.get("repair", {})
    seeds = tuple(int(seed) for seed in repair_config.get("seeds", [base_nnq_config.seed]))
    max_repair_state = int(repair_config.get("max_repair_state", 2))
    defender_actions = tuple(
        int(action) for action in repair_config.get("defender_actions", [0, 1])
    )

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "service_rate_nnq_repair_debug"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")

    seed_results = []
    for seed in seeds:
        print(f"running seed {seed}", flush=True)
        seed_nnq_config = replace(base_nnq_config, seed=seed)
        seed_evaluation_config = replace(
            evaluation_config,
            seed=evaluation_config.seed + seed,
        )
        seed_rows = _run_seed_repair(
            run_dir=run_dir,
            seed=seed,
            env=ServiceRateControlEnv(env_config),
            nnq_config=seed_nnq_config,
            evaluation_config=seed_evaluation_config,
            max_policy_grid_state=policy_grid_config.max_state,
            max_repair_state=max_repair_state,
            defender_actions=defender_actions,
        )
        best_seed_row = min(seed_rows, key=lambda row: float(row["average_cost_mean"]))
        seed_results.append(
            {
                "seed": seed,
                "rows": seed_rows,
                "best_label": best_seed_row["label"],
                "best_average_cost_mean": best_seed_row["average_cost_mean"],
            }
        )
        print(
            f"finished seed {seed}: best={best_seed_row['label']} "
            f"cost={float(best_seed_row['average_cost_mean']):.6g}",
            flush=True,
        )

    rows = _aggregate_rows(seed_results)
    write_jsonl(run_dir / "repair_results.jsonl", rows)
    write_jsonl(
        run_dir / "seed_repair_results.jsonl",
        (
            {"seed": seed_result["seed"], **row}
            for seed_result in seed_results
            for row in seed_result["rows"]
        ),
    )
    best_row = min(rows, key=lambda row: float(row["average_cost_mean"]))
    baseline_row = next(row for row in rows if row["label"] == "baseline")
    report = {
        "benchmark": "service_rate_control",
        "algorithm": "nnq_low_state_policy_repair_diagnostic",
        "run_dir": str(run_dir),
        "config": str(config_path),
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "total_steps": base_nnq_config.total_steps,
        "baseline_average_cost_mean": baseline_row["average_cost_mean"],
        "best_label": best_row["label"],
        "best_average_cost_mean": best_row["average_cost_mean"],
        "best_improvement_vs_baseline": (
            float(baseline_row["average_cost_mean"]) - float(best_row["average_cost_mean"])
        ),
        "rows": rows,
        "seed_results": seed_results,
        "interpretation": _interpretation(baseline_row, best_row),
    }
    write_json(run_dir / "summary.json", report)
    if args.json_output:
        write_json(args.json_output, report)
    if args.markdown_output:
        Path(args.markdown_output).write_text(_markdown(report), encoding="utf-8")

    print(f"wrote {run_dir}")
    print(
        "summary: "
        f"seeds={list(seeds)} "
        f"baseline={float(baseline_row['average_cost_mean']):.6g} "
        f"best={report['best_label']} "
        f"best_cost={float(best_row['average_cost_mean']):.6g} "
        f"improvement={report['best_improvement_vs_baseline']:.6g}"
    )
    return 0


def _run_seed_repair(
    run_dir: Path,
    seed: int,
    env: ServiceRateControlEnv,
    nnq_config: NNQConfig,
    evaluation_config: EvaluationConfig,
    max_policy_grid_state: int,
    max_repair_state: int,
    defender_actions: tuple[int, ...],
) -> list[dict[str, float | int | str | None]]:
    trainer = NNQTrainer(env, nnq_config)
    result = trainer.train()
    trainer.network = result.network.copy()
    trainer.target_network = result.network.copy()
    prefix = f"seed_{seed}"
    write_jsonl(run_dir / f"{prefix}_metrics.jsonl", result.metrics)

    baseline_policy = make_nnq_defender_policy(trainer)
    baseline_evaluation = evaluate_policy(
        env,
        defender_policy=baseline_policy,
        attacker_policy=random_attacker_policy,
        config=evaluation_config,
    )
    write_jsonl(run_dir / f"{prefix}_baseline_evaluation.jsonl", baseline_evaluation.rows)

    rows = []
    baseline_grid = _policy_grid_rows(
        "baseline",
        env,
        trainer,
        max_policy_grid_state,
        repair_state=None,
        repair_action=None,
    )
    write_jsonl(run_dir / f"{prefix}_baseline_policy_grid.jsonl", baseline_grid)
    rows.append(
        _result_row(
            label="baseline",
            repair_state=None,
            repair_action=None,
            evaluation_summary=baseline_evaluation.summary,
            policy_rows=baseline_grid,
        )
    )

    for repair_state in range(max_repair_state + 1):
        for repair_action in defender_actions:
            label = f"state_le_{repair_state}_action_{repair_action}"
            policy = _low_state_repair_policy(trainer, repair_state, repair_action)
            evaluation = evaluate_policy(
                env,
                defender_policy=policy,
                attacker_policy=random_attacker_policy,
                config=evaluation_config,
            )
            policy_rows = _policy_grid_rows(
                label,
                env,
                trainer,
                max_policy_grid_state,
                repair_state=repair_state,
                repair_action=repair_action,
            )
            write_jsonl(run_dir / f"{prefix}_{label}_evaluation.jsonl", evaluation.rows)
            write_jsonl(run_dir / f"{prefix}_{label}_policy_grid.jsonl", policy_rows)
            rows.append(
                _result_row(
                    label=label,
                    repair_state=repair_state,
                    repair_action=repair_action,
                    evaluation_summary=evaluation.summary,
                    policy_rows=policy_rows,
                )
            )
    return rows


def _aggregate_rows(
    seed_results: list[dict],
) -> list[dict[str, float | int | str | None]]:
    labels = [str(row["label"]) for row in seed_results[0]["rows"]]
    rows = []
    for label in labels:
        label_rows = [
            row
            for seed_result in seed_results
            for row in seed_result["rows"]
            if row["label"] == label
        ]
        first = label_rows[0]
        costs = np.array([float(row["average_cost_mean"]) for row in label_rows])
        tails = np.array([float(row["tail_fraction_mean"]) for row in label_rows])
        state0_high = np.array([float(row["state0_p_high"]) for row in label_rows])
        rows.append(
            {
                "label": label,
                "repair_state": first["repair_state"],
                "repair_action": first["repair_action"],
                "average_cost_mean": float(costs.mean()),
                "average_cost_std_across_seeds": float(costs.std(ddof=0)),
                "average_cost_min": float(costs.min()),
                "average_cost_max": float(costs.max()),
                "tail_fraction_mean": float(tails.mean()),
                "state0_p_high": float(state0_high.mean()),
                "state0_raw_nnq_p_high": float(
                    np.mean([float(row["state0_raw_nnq_p_high"]) for row in label_rows])
                ),
            }
        )
    return rows


def _low_state_repair_policy(
    trainer: NNQTrainer,
    max_repair_state: int,
    repair_action: int,
) -> DefenderPolicy:
    base_policy = make_nnq_defender_policy(trainer)

    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        if _state_load(state) <= max_repair_state:
            if repair_action not in env.defender_actions(state):
                raise ValueError("repair defender action is invalid for current state")
            return int(repair_action)
        return int(base_policy(state, rng, env))

    return policy


def _policy_grid_rows(
    label: str,
    env: ServiceRateControlEnv,
    trainer: NNQTrainer,
    max_state: int,
    repair_state: int | None,
    repair_action: int | None,
) -> list[dict[str, float | int | str]]:
    rows = []
    for state in range(max_state + 1):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        strategy = np.asarray(game["defender_strategy"], dtype=float)
        if repair_state is not None and _state_load(state) <= repair_state:
            if repair_action is None:
                raise ValueError("repair_action is required with repair_state")
            if repair_action not in (0, 1, 2):
                raise ValueError("service-rate policy grid expects defender actions 0, 1, 2")
            strategy = np.zeros(3, dtype=float)
            strategy[int(repair_action)] = 1.0
        rows.append(
            {
                "label": label,
                "state": state,
                "p_low": float(strategy[0]),
                "p_medium": float(strategy[1]),
                "p_high": float(strategy[2]),
                "nnq_raw_p_low": float(game["defender_strategy"][0]),
                "nnq_raw_p_medium": float(game["defender_strategy"][1]),
                "nnq_raw_p_high": float(game["defender_strategy"][2]),
            }
        )
    return rows


def _result_row(
    label: str,
    repair_state: int | None,
    repair_action: int | None,
    evaluation_summary: dict[str, float | int],
    policy_rows: list[dict[str, float | int | str]],
) -> dict[str, float | int | str | None]:
    state0 = next(row for row in policy_rows if int(row["state"]) == 0)
    return {
        "label": label,
        "repair_state": repair_state,
        "repair_action": repair_action,
        "average_cost_mean": float(evaluation_summary["average_cost_mean"]),
        "average_cost_std": float(evaluation_summary["average_cost_std"]),
        "discounted_cost_mean": float(evaluation_summary["discounted_cost_mean"]),
        "tail_fraction_mean": float(evaluation_summary["tail_fraction_mean"]),
        "boundary_hit_fraction_mean": float(
            evaluation_summary["boundary_hit_fraction_mean"]
        ),
        "state0_p_low": float(state0["p_low"]),
        "state0_p_medium": float(state0["p_medium"]),
        "state0_p_high": float(state0["p_high"]),
        "state0_raw_nnq_p_high": float(state0["nnq_raw_p_high"]),
    }


def _interpretation(
    baseline_row: dict[str, float | int | str | None],
    best_row: dict[str, float | int | str | None],
) -> str:
    improvement = float(baseline_row["average_cost_mean"]) - float(
        best_row["average_cost_mean"]
    )
    if improvement > 0.0 and float(best_row["state0_p_high"]) == 0.0:
        return (
            "A low-state policy repair improves NNQ rollout cost and removes "
            "state 0 high-service overuse under this diagnostic."
        )
    if improvement > 0.0:
        return "A repair improves rollout cost, but state 0 still uses high service."
    return (
        "The tested low-state policy repairs do not improve rollout cost; the NNQ "
        "failure is not fixed by this simple policy-level correction."
    )


def _markdown(report: dict) -> str:
    rows = report["rows"]
    lines = [
        "# Service-Rate NNQ Repair Diagnostic",
        "",
        f"Run dir: `{report['run_dir']}`",
        "",
        f"Baseline average cost: `{report['baseline_average_cost_mean']:.6f}`",
        f"Best repair: `{report['best_label']}`",
        f"Best average cost: `{report['best_average_cost_mean']:.6f}`",
        f"Improvement vs baseline: `{report['best_improvement_vs_baseline']:.6f}`",
        "",
        "## Candidate Results",
        "",
        "| Label | Avg cost | State 0 p_high | Tail fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{float(row['average_cost_mean']):.6f} | "
            f"{float(row['state0_p_high']):.3f} | "
            f"{float(row['tail_fraction_mean']):.6f} |"
        )
    lines.extend(["", "## Interpretation", "", str(report["interpretation"]), ""])
    return "\n".join(lines)


def _state_load(state: Hashable) -> int:
    if isinstance(state, tuple):
        return int(sum(int(value) for value in state))
    return int(state)


if __name__ == "__main__":
    raise SystemExit(main())
