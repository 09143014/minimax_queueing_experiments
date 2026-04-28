#!/usr/bin/env python3
"""Build a compact routing comparison report from existing summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALGORITHMS = ("bvi", "amq", "nnq")
MAIN_METRIC = "bvi_attacker_average_cost_mean"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-summary", required=True)
    parser.add_argument("--fitted-summary", required=True)
    parser.add_argument("--normalized-visitation-summary", required=True)
    parser.add_argument("--fitted-visitation-summary", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    report = build_report(
        normalized_summary_path=Path(args.normalized_summary),
        fitted_summary_path=Path(args.fitted_summary),
        normalized_visitation_summary_path=Path(args.normalized_visitation_summary),
        fitted_visitation_summary_path=Path(args.fitted_visitation_summary),
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


def build_report(
    *,
    normalized_summary_path: Path,
    fitted_summary_path: Path,
    normalized_visitation_summary_path: Path,
    fitted_visitation_summary_path: Path,
) -> dict[str, Any]:
    normalized_summary = _read_json(normalized_summary_path)
    fitted_summary = _read_json(fitted_summary_path)
    normalized_visitation = _read_json(normalized_visitation_summary_path)
    fitted_visitation = _read_json(fitted_visitation_summary_path)

    normalized = _comparison_block(
        "normalized_amq",
        normalized_summary_path,
        normalized_summary,
        amq_label="normalized AMQ",
    )
    fitted = _comparison_block(
        "fitted_amq",
        fitted_summary_path,
        fitted_summary,
        amq_label="fitted AMQ",
    )
    normalized_amq_mean = normalized["bvi_attacker_average_cost"]["amq"]
    fitted_amq_mean = fitted["bvi_attacker_average_cost"]["amq"]
    nnq_mean = normalized["bvi_attacker_average_cost"]["nnq"]
    return {
        "benchmark": "routing",
        "main_metric": MAIN_METRIC,
        "runs": {
            "normalized_amq": normalized,
            "fitted_amq": fitted,
        },
        "visitation": {
            "normalized_amq": _visitation_block(
                normalized_visitation_summary_path,
                normalized_visitation,
            ),
            "fitted_amq": _visitation_block(
                fitted_visitation_summary_path,
                fitted_visitation,
            ),
        },
        "conclusion": {
            "main_amq_performance_baseline": "normalized_amq",
            "diagnostic_calibration_baseline": "fitted_amq",
            "normalized_amq_margin_vs_nnq": nnq_mean - normalized_amq_mean,
            "fitted_amq_margin_vs_nnq": nnq_mean - fitted_amq_mean,
            "fitted_amq_cost_delta_vs_normalized": (
                fitted_amq_mean - normalized_amq_mean
            ),
            "summary": (
                "normalized AMQ is the current performance baseline; fitted "
                "AMQ improves policy sanity but does not improve the main "
                "BVI-attacker cost."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    normalized = report["runs"]["normalized_amq"]
    fitted = report["runs"]["fitted_amq"]
    normalized_visitation = report["visitation"]["normalized_amq"]
    fitted_visitation = report["visitation"]["fitted_amq"]
    conclusion = report["conclusion"]
    lines = [
        "# Routing Comparison Report",
        "",
        "## Goal",
        "",
        "Build a credible routing benchmark for BVI / AMQ / NNQ under shared "
        "environment settings, seeds, and evaluation policies. The main metric "
        "is average cost under the common BVI attacker.",
        "",
        "## Compared Runs",
        "",
        _comparison_markdown(
            "Normalized AMQ, 3 seeds, 50 eval episodes",
            normalized,
        ),
        "",
        _comparison_markdown(
            "Fitted-Calibrated Normalized AMQ, 3 seeds, 50 eval episodes",
            fitted,
        ),
        "",
        "## AMQ Policy Sanity Diagnostics",
        "",
        _visitation_markdown("Before fitted calibration", normalized_visitation),
        "",
        _visitation_markdown("After fitted calibration", fitted_visitation),
        "",
        "## Current Defensible Baselines",
        "",
        f"Main AMQ performance baseline: `{conclusion['main_amq_performance_baseline']}`",
        "",
        f"Diagnostic AMQ calibration baseline: `{conclusion['diagnostic_calibration_baseline']}`",
        "",
        (
            "Normalized AMQ beats NNQ on the main metric by "
            f"{_format_float(conclusion['normalized_amq_margin_vs_nnq'])}. "
            "Fitted AMQ beats NNQ by "
            f"{_format_float(conclusion['fitted_amq_margin_vs_nnq'])}, but is worse "
            "than normalized AMQ by "
            f"{_format_float(conclusion['fitted_amq_cost_delta_vs_normalized'])}."
        ),
        "",
        conclusion["summary"],
        "",
    ]
    return "\n".join(lines)


def _comparison_block(
    key: str,
    source: Path,
    summary: dict[str, Any],
    *,
    amq_label: str,
) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    costs = {
        algorithm: float(aggregate[algorithm][MAIN_METRIC]["mean"])
        for algorithm in ALGORITHMS
    }
    return {
        "key": key,
        "amq_label": amq_label,
        "source": str(source),
        "bvi_attacker_average_cost": costs,
        "bvi_attacker_ranking_counts": summary["ranking_counts"]["bvi_attacker"],
        "per_seed_bvi_attacker_average_cost": _per_seed_costs(summary),
    }


def _visitation_block(source: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(source),
        "num_states": int(summary["num_states"]),
        "num_visited_states": int(summary["num_visited_states"]),
        "num_visited_over_defend_states": int(
            summary["num_visited_over_defend_states"]
        ),
        "visit_weighted_abs_gap_mean": float(
            summary["visit_weighted_abs_gap_mean"]
        ),
        "visited_abs_gap_mean": float(summary["visited_abs_gap_mean"]),
        "unvisited_abs_gap_mean": float(summary["unvisited_abs_gap_mean"]),
    }


def _per_seed_costs(summary: dict[str, Any]) -> list[dict[str, float | int]]:
    rows = []
    for row in summary["rows"]:
        item: dict[str, float | int] = {"seed": int(row["seed"])}
        for algorithm_row in row["algorithm_rows"]:
            algorithm = algorithm_row["algorithm"]
            if algorithm in ALGORITHMS:
                item[algorithm] = float(algorithm_row[MAIN_METRIC])
        rows.append(item)
    return rows


def _comparison_markdown(title: str, block: dict[str, Any]) -> str:
    lines = [
        f"### {title}",
        "",
        "Source:",
        "",
        f"`{block['source']}`",
        "",
        "Common BVI-attacker average cost:",
        "",
        "| Algorithm | Mean |",
        "| --- | ---: |",
    ]
    labels = {
        "bvi": "BVI",
        "amq": block["amq_label"],
        "nnq": "NNQ",
    }
    for algorithm in ALGORITHMS:
        lines.append(
            f"| {labels[algorithm]} | "
            f"{_format_float(block['bvi_attacker_average_cost'][algorithm])} |"
        )
    lines.extend(
        [
            "",
            "Ranking counts under common BVI attacker:",
            "",
            "| Algorithm | Best-count |",
            "| --- | ---: |",
        ]
    )
    for algorithm in ALGORITHMS:
        lines.append(
            f"| {labels[algorithm]} | "
            f"{block['bvi_attacker_ranking_counts'][algorithm]} |"
        )
    lines.extend(
        [
            "",
            "Per-seed common BVI-attacker average cost:",
            "",
            "| Seed | BVI | AMQ | NNQ |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in block["per_seed_bvi_attacker_average_cost"]:
        lines.append(
            f"| {row['seed']} | {_format_float(row['bvi'])} | "
            f"{_format_float(row['amq'])} | {_format_float(row['nnq'])} |"
        )
    return "\n".join(lines)


def _visitation_markdown(title: str, block: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {title}",
            "",
            "Source:",
            "",
            f"`{block['source']}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            (
                f"| visited states | {block['num_visited_states']} / "
                f"{block['num_states']} |"
            ),
            (
                "| visited over-defend states | "
                f"{block['num_visited_over_defend_states']} |"
            ),
            (
                "| visit-weighted policy gap | "
                f"{_format_float(block['visit_weighted_abs_gap_mean'])} |"
            ),
            (
                "| visited abs gap mean | "
                f"{_format_float(block['visited_abs_gap_mean'])} |"
            ),
            (
                "| unvisited abs gap mean | "
                f"{_format_float(block['unvisited_abs_gap_mean'])} |"
            ),
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
