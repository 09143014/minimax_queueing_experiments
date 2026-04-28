#!/usr/bin/env python3
"""Compare AMQ policy gaps on states visited under a common BVI attacker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-summary", required=True)
    parser.add_argument("--fitted-summary", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed subset, e.g. 0,1.",
    )
    args = parser.parse_args()

    normalized_summary_path = Path(args.normalized_summary)
    fitted_summary_path = Path(args.fitted_summary)
    seed_filter = _parse_seed_filter(args.seeds)
    report = build_diagnostic(
        normalized_summary_path=normalized_summary_path,
        fitted_summary_path=fitted_summary_path,
        seed_filter=seed_filter,
    )

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_output}")
    print(f"wrote {markdown_output}")
    return 0


def build_diagnostic(
    *,
    normalized_summary_path: Path,
    fitted_summary_path: Path,
    seed_filter: set[int] | None = None,
) -> dict[str, Any]:
    normalized = _read_json(normalized_summary_path)
    fitted = _read_json(fitted_summary_path)
    normalized_rows = _amq_rows_by_seed(normalized)
    fitted_rows = _amq_rows_by_seed(fitted)
    seeds = sorted(set(normalized_rows) & set(fitted_rows))
    if seed_filter is not None:
        seeds = [seed for seed in seeds if seed in seed_filter]
    if not seeds:
        raise ValueError("no matching seeds to diagnose")

    seed_reports = []
    for seed in seeds:
        print(f"diagnosing seed {seed}", flush=True)
        normalized_seed = _variant_seed_diagnostic(
            "normalized_amq",
            seed,
            Path(normalized_rows[seed]["config"]),
        )
        fitted_seed = _variant_seed_diagnostic(
            "fitted_amq",
            seed,
            Path(fitted_rows[seed]["config"]),
        )
        seed_reports.append(
            {
                "seed": seed,
                "normalized_amq": normalized_seed,
                "fitted_amq": fitted_seed,
                "delta_fitted_minus_normalized": _summary_delta(
                    normalized_seed["summary"],
                    fitted_seed["summary"],
                ),
            }
        )

    return {
        "benchmark": "routing",
        "question": (
            "Does fitted calibration reduce full-grid policy gap while losing "
            "defense mass on BVI-attacker visited states?"
        ),
        "sources": {
            "normalized_summary": str(normalized_summary_path),
            "fitted_summary": str(fitted_summary_path),
        },
        "seeds": seeds,
        "seed_diagnostics": seed_reports,
        "aggregate": _aggregate(seed_reports),
        "interpretation": _interpret(seed_reports),
    }


def summarize_weighted_policy_gap(
    comparison_rows: list[dict[str, Any]],
    visitation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visits = {
        tuple(row["state"]) if isinstance(row["state"], list) else row["state"]: row
        for row in visitation_rows
    }
    joined = []
    for row in comparison_rows:
        state = tuple(row["state"]) if isinstance(row["state"], list) else row["state"]
        visit = visits.get(state, {"visit_count": 0, "visit_fraction": 0.0})
        joined.append(
            {
                **row,
                "visit_count": int(visit["visit_count"]),
                "visit_fraction": float(visit["visit_fraction"]),
                "weighted_abs_gap": float(row["p_defend_abs_gap"])
                * float(visit["visit_fraction"]),
                "weighted_signed_gap": float(row["p_defend_signed_gap"])
                * float(visit["visit_fraction"]),
            }
        )

    visited = [row for row in joined if row["visit_count"] > 0]
    visited_over = [row for row in visited if row["amq_over_defends"]]
    visited_under = [row for row in visited if row["amq_under_defends"]]
    return {
        "num_states": len(joined),
        "num_visited_states": len(visited),
        "num_visited_over_defend_states": len(visited_over),
        "num_visited_under_defend_states": len(visited_under),
        "visit_weighted_abs_gap_mean": sum(
            row["weighted_abs_gap"] for row in joined
        ),
        "visit_weighted_signed_gap_mean": sum(
            row["weighted_signed_gap"] for row in joined
        ),
        "visited_abs_gap_mean": _mean(visited, "p_defend_abs_gap"),
        "visited_signed_gap_mean": _mean(visited, "p_defend_signed_gap"),
        "visited_p_defend_amq_mean": _weighted_mean(
            visited,
            "p_defend_amq",
            "visit_fraction",
        ),
        "visited_p_defend_bvi_reference_mean": _weighted_mean(
            visited,
            "p_defend_bvi_reference",
            "visit_fraction",
        ),
        "top_visited_gap_states": [
            {
                "state": row["state"],
                "visit_count": row["visit_count"],
                "visit_fraction": row["visit_fraction"],
                "p_defend_amq": row["p_defend_amq"],
                "p_defend_bvi_reference": row["p_defend_bvi_reference"],
                "p_defend_signed_gap": row["p_defend_signed_gap"],
                "weighted_abs_gap": row["weighted_abs_gap"],
            }
            for row in sorted(
                visited,
                key=lambda item: (
                    item["weighted_abs_gap"],
                    item["visit_count"],
                ),
                reverse=True,
            )[:10]
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Routing BVI-Attacker Visitation Diagnostic",
        "",
        "## Question",
        "",
        report["question"],
        "",
        "## Aggregate Delta",
        "",
        "Delta is fitted AMQ minus normalized AMQ. Negative signed gap means fitted AMQ defends less on BVI-attacker visited states.",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        (
            "| visit-weighted abs gap | "
            f"{_format_float(aggregate['mean_visit_weighted_abs_gap_delta'])} |"
        ),
        (
            "| visit-weighted signed gap | "
            f"{_format_float(aggregate['mean_visit_weighted_signed_gap_delta'])} |"
        ),
        (
            "| visited AMQ defend probability | "
            f"{_format_float(aggregate['mean_visited_p_defend_amq_delta'])} |"
        ),
        (
            "| visited BVI defend probability | "
            f"{_format_float(aggregate['mean_visited_p_defend_bvi_delta'])} |"
        ),
        (
            "| visited over-defend states | "
            f"{_format_float(aggregate['mean_visited_over_defend_states_delta'])} |"
        ),
        "",
        "## Per-Seed Summary",
        "",
        "| Seed | abs gap delta | signed gap delta | AMQ defend delta | BVI defend delta | over-defend delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_diagnostics"]:
        delta = row["delta_fitted_minus_normalized"]
        lines.append(
            f"| {row['seed']} | "
            f"{_format_float(delta['visit_weighted_abs_gap_mean'])} | "
            f"{_format_float(delta['visit_weighted_signed_gap_mean'])} | "
            f"{_format_float(delta['visited_p_defend_amq_mean'])} | "
            f"{_format_float(delta['visited_p_defend_bvi_reference_mean'])} | "
            f"{_format_float(delta['num_visited_over_defend_states'])} |"
        )
    lines.extend(["", "## Top Fitted Visited Gap States", ""])
    for row in report["seed_diagnostics"]:
        lines.extend(
            [
                f"### Seed {row['seed']}",
                "",
                "| Variant | State | Visits | AMQ defend | BVI defend | Signed gap | Weighted abs gap |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in ("normalized_amq", "fitted_amq"):
            for item in row[variant]["summary"]["top_visited_gap_states"][:5]:
                lines.append(
                    f"| {variant} | {item['state']} | {item['visit_count']} | "
                    f"{_format_float(item['p_defend_amq'])} | "
                    f"{_format_float(item['p_defend_bvi_reference'])} | "
                    f"{_format_float(item['p_defend_signed_gap'])} | "
                    f"{_format_float(item['weighted_abs_gap'])} |"
                )
        lines.append("")
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def _variant_seed_diagnostic(
    variant: str,
    seed: int,
    config_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    env_config = build_routing_config(config)
    env = RoutingEnv(env_config)
    amq_config = build_amq_config(config)
    evaluation_config = build_evaluation_config(config)
    policy_grid_config = build_policy_grid_config(config)
    bvi_config = config["bvi"]
    max_queue_length = int(bvi_config["max_queue_length"])

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
    comparison_rows, _comparison_summary = compare_amq_bvi_routing_policies(
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
    return {
        "variant": variant,
        "seed": seed,
        "config": str(config_path),
        "summary": summarize_weighted_policy_gap(comparison_rows, visitation_rows),
    }


def _aggregate(seed_reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = {
        "visit_weighted_abs_gap_mean": "mean_visit_weighted_abs_gap_delta",
        "visit_weighted_signed_gap_mean": "mean_visit_weighted_signed_gap_delta",
        "visited_p_defend_amq_mean": "mean_visited_p_defend_amq_delta",
        "visited_p_defend_bvi_reference_mean": "mean_visited_p_defend_bvi_delta",
        "num_visited_over_defend_states": "mean_visited_over_defend_states_delta",
    }
    return {
        output_key: sum(
            row["delta_fitted_minus_normalized"][key] for row in seed_reports
        )
        / len(seed_reports)
        for key, output_key in keys.items()
    }


def _summary_delta(
    normalized: dict[str, Any],
    fitted: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "num_visited_states",
        "num_visited_over_defend_states",
        "num_visited_under_defend_states",
        "visit_weighted_abs_gap_mean",
        "visit_weighted_signed_gap_mean",
        "visited_abs_gap_mean",
        "visited_signed_gap_mean",
        "visited_p_defend_amq_mean",
        "visited_p_defend_bvi_reference_mean",
    )
    return {key: float(fitted[key]) - float(normalized[key]) for key in keys}


def _interpret(seed_reports: list[dict[str, Any]]) -> str:
    signed_deltas = [
        row["delta_fitted_minus_normalized"]["visit_weighted_signed_gap_mean"]
        for row in seed_reports
    ]
    defend_deltas = [
        row["delta_fitted_minus_normalized"]["visited_p_defend_amq_mean"]
        for row in seed_reports
    ]
    if all(delta < 0 for delta in defend_deltas):
        return (
            "Under the common BVI attacker, fitted calibration lowers AMQ's "
            "visited-state defense probability on every diagnosed seed. This "
            "supports the cost-loss hypothesis: fitted calibration removes "
            "global over-defense, but under BVI-attacker visitation it also "
            "removes defense mass that had been protecting cost-relevant states."
        )
    if sum(signed_deltas) < 0:
        return (
            "The BVI-attacker-weighted signed gap moves downward on average. "
            "Inspect seeds where defense probability does not fall before "
            "changing calibration parameters."
        )
    return (
        "The BVI-attacker-weighted visitation diagnostic does not support a "
        "simple defense-mass-loss explanation. Inspect top visited states before "
        "running another calibration experiment."
    )


def _amq_rows_by_seed(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for seed_row in summary["rows"]:
        seed = int(seed_row["seed"])
        for algorithm_row in seed_row["algorithm_rows"]:
            if algorithm_row["algorithm"] == "amq":
                rows[seed] = algorithm_row
                break
        else:
            raise ValueError(f"seed {seed} is missing AMQ row")
    return rows


def _parse_seed_filter(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    weight_sum = sum(float(row[weight_key]) for row in rows)
    if weight_sum == 0.0:
        return 0.0
    return float(
        sum(float(row[value_key]) * float(row[weight_key]) for row in rows)
        / weight_sum
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


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
