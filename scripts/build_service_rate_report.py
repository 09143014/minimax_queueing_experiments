#!/usr/bin/env python3
"""Build a compact service-rate-control comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("bvi", "amq", "nnq")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    report = build_report(Path(args.summary))
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


def build_report(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    aggregate = summary["aggregate"]
    costs = {
        method: float(aggregate[method]["average_cost_mean"]["mean"])
        for method in METHODS
    }
    thresholds = {
        method: aggregate[method].get("first_state_p_high_at_least_threshold", {}).get(
            "mean"
        )
        for method in METHODS
    }
    return {
        "benchmark": "service_rate_control",
        "source": str(summary_path),
        "num_seeds": int(summary["num_seeds"]),
        "seeds": summary["seeds"],
        "average_cost_mean": costs,
        "ranking_counts": summary["ranking_counts"]["average_cost"],
        "first_state_p_high_at_least_threshold_mean": thresholds,
        "per_seed_average_cost": _per_seed_costs(summary),
        "conclusion": {
            "selected_amq_baseline": "amq",
            "amq_margin_vs_nnq": costs["nnq"] - costs["amq"],
            "amq_gap_vs_bvi": costs["amq"] - costs["bvi"],
            "summary": (
                "BVI is the strongest method on this service-rate-control debug "
                "comparison. AMQ is consistently better than NNQ, but remains "
                "behind BVI."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Service-Rate-Control Report",
        "",
        "## Source",
        "",
        f"`{report['source']}`",
        "",
        "## Main Result",
        "",
        "Average cost, 3 seeds:",
        "",
        "| Method | Mean cost | Best-count | First high-service state |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {"bvi": "BVI", "amq": "AMQ", "nnq": "NNQ"}
    for method in METHODS:
        threshold = report["first_state_p_high_at_least_threshold_mean"][method]
        threshold_text = "n/a" if threshold is None else _format_float(float(threshold))
        lines.append(
            f"| {labels[method]} | "
            f"{_format_float(report['average_cost_mean'][method])} | "
            f"{report['ranking_counts'][method]} | {threshold_text} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Average Cost",
            "",
            "| Seed | BVI | AMQ | NNQ |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["per_seed_average_cost"]:
        lines.append(
            f"| {row['seed']} | {_format_float(row['bvi'])} | "
            f"{_format_float(row['amq'])} | {_format_float(row['nnq'])} |"
        )
    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"AMQ beats NNQ by {_format_float(conclusion['amq_margin_vs_nnq'])} "
                "average cost, but trails BVI by "
                f"{_format_float(conclusion['amq_gap_vs_bvi'])}."
            ),
            "",
            conclusion["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def _per_seed_costs(summary: dict[str, Any]) -> list[dict[str, float | int]]:
    rows = []
    for row in summary["rows"]:
        item: dict[str, float | int] = {"seed": int(row["seed"])}
        for method_row in row["method_rows"]:
            method = method_row["method"]
            if method in METHODS:
                item[method] = float(method_row["average_cost_mean"])
        rows.append(item)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
