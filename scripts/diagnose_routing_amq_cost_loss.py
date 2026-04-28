#!/usr/bin/env python3
"""Diagnose fitted AMQ cost loss against normalized AMQ from existing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AMQ_METRICS = (
    "average_cost_mean",
    "always_attack_average_cost_mean",
    "minimax_average_cost_mean",
    "bvi_attacker_average_cost_mean",
    "policy_gap_mean",
    "q_reference_gap_mean",
    "defend_probability_mean",
    "defend_states",
)
BUCKET_METRICS = (
    "p_defend_abs_gap_mean",
    "p_defend_signed_gap_mean",
    "q_reference_abs_gap_mean",
    "amq_bellman_abs_residual_mean",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-summary", required=True)
    parser.add_argument("--fitted-summary", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    report = build_diagnostic(
        normalized_summary_path=Path(args.normalized_summary),
        fitted_summary_path=Path(args.fitted_summary),
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
) -> dict[str, Any]:
    normalized = _read_json(normalized_summary_path)
    fitted = _read_json(fitted_summary_path)
    normalized_by_seed = _amq_rows_by_seed(normalized)
    fitted_by_seed = _amq_rows_by_seed(fitted)
    seeds = sorted(set(normalized_by_seed) & set(fitted_by_seed))
    seed_rows = [
        _seed_diagnostic(seed, normalized_by_seed[seed], fitted_by_seed[seed])
        for seed in seeds
    ]
    deltas = [row["metric_deltas"]["bvi_attacker_average_cost_mean"] for row in seed_rows]
    worsened = [row for row in seed_rows if row["metric_deltas"]["bvi_attacker_average_cost_mean"] > 0]
    improved = [row for row in seed_rows if row["metric_deltas"]["bvi_attacker_average_cost_mean"] < 0]
    mean_delta = sum(deltas) / len(deltas)
    return {
        "benchmark": "routing",
        "question": (
            "Why does fitted AMQ improve policy-gap diagnostics but lose "
            "BVI-attacker rollout cost against normalized AMQ?"
        ),
        "sources": {
            "normalized_summary": str(normalized_summary_path),
            "fitted_summary": str(fitted_summary_path),
        },
        "num_seeds": len(seeds),
        "seed_diagnostics": seed_rows,
        "aggregate": {
            "mean_bvi_attacker_cost_delta": mean_delta,
            "mean_average_cost_delta": _mean_delta(seed_rows, "average_cost_mean"),
            "mean_always_attack_cost_delta": _mean_delta(
                seed_rows,
                "always_attack_average_cost_mean",
            ),
            "mean_minimax_cost_delta": _mean_delta(
                seed_rows,
                "minimax_average_cost_mean",
            ),
            "num_cost_worse_seeds": len(worsened),
            "num_cost_better_seeds": len(improved),
            "largest_cost_loss_seed": max(
                seed_rows,
                key=lambda row: row["metric_deltas"]["bvi_attacker_average_cost_mean"],
            )["seed"] if worsened else None,
            "largest_cost_gain_seed": min(
                improved,
                key=lambda row: row["metric_deltas"]["bvi_attacker_average_cost_mean"],
            )["seed"] if improved else None,
            "mean_policy_gap_delta": _mean_delta(seed_rows, "policy_gap_mean"),
            "mean_q_reference_gap_delta": _mean_delta(seed_rows, "q_reference_gap_mean"),
            "mean_defend_probability_delta": _mean_delta(
                seed_rows,
                "defend_probability_mean",
            ),
            "mean_defend_states_delta": _mean_delta(seed_rows, "defend_states"),
        },
        "interpretation": _interpret(seed_rows),
    }


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Routing AMQ Cost-Loss Diagnostic",
        "",
        "## Question",
        "",
        report["question"],
        "",
        "## Sources",
        "",
        f"- normalized: `{report['sources']['normalized_summary']}`",
        f"- fitted: `{report['sources']['fitted_summary']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Delta fitted - normalized |",
        "| --- | ---: |",
        (
            "| BVI-attacker average cost mean | "
            f"{_format_float(aggregate['mean_bvi_attacker_cost_delta'])} |"
        ),
        f"| random-attacker average cost mean | {_format_float(aggregate['mean_average_cost_delta'])} |",
        (
            "| always-attack average cost mean | "
            f"{_format_float(aggregate['mean_always_attack_cost_delta'])} |"
        ),
        f"| minimax average cost mean | {_format_float(aggregate['mean_minimax_cost_delta'])} |",
        f"| policy gap mean | {_format_float(aggregate['mean_policy_gap_delta'])} |",
        (
            "| q-reference gap mean | "
            f"{_format_float(aggregate['mean_q_reference_gap_delta'])} |"
        ),
        (
            "| defend probability mean | "
            f"{_format_float(aggregate['mean_defend_probability_delta'])} |"
        ),
        f"| defend states | {_format_float(aggregate['mean_defend_states_delta'])} |",
        "",
        (
            f"Cost worsened on {aggregate['num_cost_worse_seeds']} seeds and improved "
            f"on {aggregate['num_cost_better_seeds']} seeds under the common "
            f"BVI attacker. Largest loss seed: {aggregate['largest_cost_loss_seed']}; "
            f"largest gain seed: {aggregate['largest_cost_gain_seed']}."
        ),
        "",
        "## Per-Seed AMQ Deltas",
        "",
        "| Seed | BVI-attacker cost | policy gap | q gap | defend prob | defend states |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_diagnostics"]:
        deltas = row["metric_deltas"]
        lines.append(
            f"| {row['seed']} | "
            f"{_format_float(deltas['bvi_attacker_average_cost_mean'])} | "
            f"{_format_float(deltas['policy_gap_mean'])} | "
            f"{_format_float(deltas['q_reference_gap_mean'])} | "
            f"{_format_float(deltas['defend_probability_mean'])} | "
            f"{_format_float(deltas['defend_states'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest Bucket Changes",
            "",
        ]
    )
    for row in report["seed_diagnostics"]:
        lines.extend(
            [
                f"### Seed {row['seed']}",
                "",
                "| Bucket | Metric | Delta |",
                "| --- | --- | ---: |",
            ]
        )
        for item in row["largest_bucket_deltas"]:
            lines.append(
                f"| {item['bucket']}={item['bucket_value']} | {item['metric']} | "
                f"{_format_float(item['delta'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def _seed_diagnostic(
    seed: int,
    normalized_row: dict[str, Any],
    fitted_row: dict[str, Any],
) -> dict[str, Any]:
    normalized_summary = _read_json(Path(normalized_row["run_dir"]) / "summary.json")
    fitted_summary = _read_json(Path(fitted_row["run_dir"]) / "summary.json")
    return {
        "seed": seed,
        "normalized": _metric_block(normalized_row),
        "fitted": _metric_block(fitted_row),
        "metric_deltas": _metric_deltas(normalized_row, fitted_row),
        "largest_bucket_deltas": _largest_bucket_deltas(
            normalized_summary,
            fitted_summary,
        ),
    }


def _metric_block(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in AMQ_METRICS
        if row.get(key) is not None
    }


def _metric_deltas(
    normalized_row: dict[str, Any],
    fitted_row: dict[str, Any],
) -> dict[str, float]:
    return {
        key: float(fitted_row[key]) - float(normalized_row[key])
        for key in AMQ_METRICS
        if normalized_row.get(key) is not None and fitted_row.get(key) is not None
    }


def _largest_bucket_deltas(
    normalized_summary: dict[str, Any],
    fitted_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for bucket_name, key in (
        ("total_queue", "total_queue"),
        ("imbalance", "imbalance"),
    ):
        rows.extend(
            _bucket_deltas(
                bucket_name,
                key,
                normalized_summary.get("policy_comparison", {}),
                fitted_summary.get("policy_comparison", {}),
            )
        )
        rows.extend(
            _bucket_deltas(
                bucket_name,
                key,
                normalized_summary.get("q_diagnostic", {}),
                fitted_summary.get("q_diagnostic", {}),
            )
        )
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows[:8]


def _bucket_deltas(
    bucket_name: str,
    key: str,
    normalized_section: dict[str, Any],
    fitted_section: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized_buckets = {
        item[key]: item
        for item in normalized_section.get(f"by_{bucket_name}", [])
        if key in item
    }
    fitted_buckets = {
        item[key]: item
        for item in fitted_section.get(f"by_{bucket_name}", [])
        if key in item
    }
    rows = []
    for bucket_value in sorted(set(normalized_buckets) & set(fitted_buckets)):
        normalized_item = normalized_buckets[bucket_value]
        fitted_item = fitted_buckets[bucket_value]
        for metric in BUCKET_METRICS:
            if metric in normalized_item and metric in fitted_item:
                rows.append(
                    {
                        "bucket": bucket_name,
                        "bucket_value": bucket_value,
                        "metric": metric,
                        "normalized": float(normalized_item[metric]),
                        "fitted": float(fitted_item[metric]),
                        "delta": float(fitted_item[metric])
                        - float(normalized_item[metric]),
                    }
                )
    return rows


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


def _mean_delta(seed_rows: list[dict[str, Any]], metric: str) -> float:
    values = [row["metric_deltas"][metric] for row in seed_rows]
    return sum(values) / len(values)


def _interpret(seed_rows: list[dict[str, Any]]) -> str:
    cost_deltas = [
        row["metric_deltas"]["bvi_attacker_average_cost_mean"]
        for row in seed_rows
    ]
    always_attack_deltas = [
        row["metric_deltas"]["always_attack_average_cost_mean"]
        for row in seed_rows
    ]
    policy_deltas = [row["metric_deltas"]["policy_gap_mean"] for row in seed_rows]
    defend_probability_deltas = [
        row["metric_deltas"]["defend_probability_mean"]
        for row in seed_rows
    ]
    if (
        all(delta > 0 for delta in cost_deltas)
        and all(delta < 0 for delta in always_attack_deltas)
        and all(delta < 0 for delta in defend_probability_deltas)
    ):
        return (
            "The cost loss is specific to the common BVI attacker: fitted AMQ "
            "slightly improves always-attack cost on every seed, while BVI-attacker "
            "cost worsens on every seed. Fitted calibration also lowers defense "
            "probability on every seed. The likely mechanism is that calibration "
            "removes broad over-defense, which improves policy-shape diagnostics, "
            "but also removes defense mass that was useful against the BVI "
            "attacker's state-action visitation. The next diagnostic should be a "
            "visited-state, BVI-attacker-weighted policy-gap comparison, not blind "
            "eta/pass tuning."
        )
    if all(delta < 0 for delta in policy_deltas) and sum(cost_deltas) > 0:
        return (
            "Fitted calibration reduces global policy gap on every seed, but "
            "the BVI-attacker rollout cost still increases on average. The "
            "dominant aggregate change is a large drop in defend probability, "
            "so the likely failure mode is not residual over-defense alone; "
            "calibration appears to remove useful defense mass on some "
            "cost-relevant trajectories, especially where fitted cost worsens."
        )
    if sum(defend_probability_deltas) < 0:
        return (
            "Fitted calibration lowers defense probability overall. Inspect "
            "the largest-loss seed before changing learning rates."
        )
    return (
        "The aggregate deltas do not point to a single monotone mechanism. "
        "Inspect the per-seed and bucket deltas before running new experiments."
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
