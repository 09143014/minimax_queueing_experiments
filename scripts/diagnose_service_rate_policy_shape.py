#!/usr/bin/env python3
"""Diagnose service-rate policy shapes from an existing multiseed comparison."""

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

    report = build_diagnostic(Path(args.summary))
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


def build_diagnostic(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    seed_rows = []
    for row in summary["rows"]:
        seed_rows.append(_seed_diagnostic(int(row["seed"]), row["method_rows"]))
    aggregate = _aggregate(seed_rows)
    return {
        "benchmark": "service_rate_control",
        "source": str(summary_path),
        "seeds": summary["seeds"],
        "seed_diagnostics": seed_rows,
        "aggregate": aggregate,
        "interpretation": _interpret(aggregate),
    }


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Service-Rate Policy-Shape Diagnostic",
        "",
        "## Source",
        "",
        f"`{report['source']}`",
        "",
        "## Aggregate Policy Shape",
        "",
        "| Method | mean p_high(state=0) | mean p_medium(state=0) | mean p_low(state=0) | mean first high state |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"bvi": "BVI", "amq": "AMQ", "nnq": "NNQ"}
    for method in METHODS:
        method_row = aggregate[method]
        lines.append(
            f"| {labels[method]} | "
            f"{_format_float(method_row['mean_p_high_state_0'])} | "
            f"{_format_float(method_row['mean_p_medium_state_0'])} | "
            f"{_format_float(method_row['mean_p_low_state_0'])} | "
            f"{_format_float(method_row['mean_first_high_state'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed State 0 Policy",
            "",
            "| Seed | Method | p_low | p_medium | p_high | average cost |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed_row in report["seed_diagnostics"]:
        for method in METHODS:
            row = seed_row[method]
            lines.append(
                f"| {seed_row['seed']} | {labels[method]} | "
                f"{_format_float(row['state_0']['p_low'])} | "
                f"{_format_float(row['state_0']['p_medium'])} | "
                f"{_format_float(row['state_0']['p_high'])} | "
                f"{_format_float(row['average_cost_mean'])} |"
            )
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def _seed_diagnostic(seed: int, method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed}
    for method_row in method_rows:
        method = method_row["method"]
        if method in METHODS:
            grid = _read_policy_grid(Path(method_row["run_dir"]) / "policy_grid.jsonl")
            row[method] = {
                "average_cost_mean": float(method_row["average_cost_mean"]),
                "first_state_p_high_at_least_threshold": method_row.get(
                    "first_state_p_high_at_least_threshold"
                ),
                "state_0": grid[0],
            }
    return row


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate = {}
    for method in METHODS:
        method_rows = [row[method] for row in seed_rows]
        aggregate[method] = {
            "mean_p_high_state_0": _mean(
                method_rows,
                lambda row: row["state_0"]["p_high"],
            ),
            "mean_p_medium_state_0": _mean(
                method_rows,
                lambda row: row["state_0"]["p_medium"],
            ),
            "mean_p_low_state_0": _mean(
                method_rows,
                lambda row: row["state_0"]["p_low"],
            ),
            "mean_first_high_state": _mean(
                method_rows,
                lambda row: row["first_state_p_high_at_least_threshold"],
            ),
            "mean_average_cost": _mean(
                method_rows,
                lambda row: row["average_cost_mean"],
            ),
        }
    return aggregate


def _interpret(aggregate: dict[str, dict[str, float]]) -> str:
    nnq_high_0 = aggregate["nnq"]["mean_p_high_state_0"]
    bvi_high_0 = aggregate["bvi"]["mean_p_high_state_0"]
    if nnq_high_0 > 0.5 and bvi_high_0 < 0.5:
        return (
            "NNQ's weak service-rate result appears to be an over-service policy "
            "shape problem: it assigns high service probability at state 0, while "
            "BVI avoids high service at the empty queue. The next NNQ step should "
            "target policy calibration or budget/architecture, not environment "
            "changes."
        )
    return (
        "The policy-shape diagnostic does not isolate a clear empty-state "
        "over-service issue. Inspect per-seed grids before changing NNQ."
    )


def _read_policy_grid(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows[int(item["state"])] = {
                "p_low": float(item["p_low"]),
                "p_medium": float(item["p_medium"]),
                "p_high": float(item["p_high"]),
            }
    return rows


def _mean(rows: list[dict[str, Any]], fn) -> float:
    values = [fn(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
