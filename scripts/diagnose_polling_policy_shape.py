#!/usr/bin/env python3
"""Diagnose polling defender-policy shapes from a comparison run."""

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
    method_rows = []
    for row in summary["rows"]:
        method = str(row["method"])
        if method in METHODS:
            method_rows.append(_method_diagnostic(method, row))
    aggregate = {row["method"]: row for row in method_rows}
    return {
        "benchmark": "polling",
        "source": str(summary_path),
        "methods": [row["method"] for row in method_rows],
        "method_diagnostics": method_rows,
        "aggregate": aggregate,
        "interpretation": _interpret(aggregate),
    }


def render_markdown(report: dict[str, Any]) -> str:
    labels = {"bvi": "BVI", "amq": "AMQ", "nnq": "NNQ"}
    lines = [
        "# Polling Policy-Shape Diagnostic",
        "",
        "## Source",
        "",
        f"`{report['source']}`",
        "",
        "## Aggregate Policy Shape",
        "",
        "| Method | Avg cost | Mean p_defend | Defend states | Gap defend states | Classification |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in METHODS:
        row = report["aggregate"][method]
        lines.append(
            f"| {labels[method]} | "
            f"{_format_float(row['average_cost_mean'])} | "
            f"{_format_float(row['defend_probability_mean'])} | "
            f"{row['num_states_p_defend_at_least_threshold']} / {row['num_policy_states']} | "
            f"{row['num_gap_states_p_defend_at_least_threshold']} / {row['num_gap_states']} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## By Queue Gap",
            "",
            "| Method | Queue gap | States | Mean p_defend | Max p_defend |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        for group in report["aggregate"][method]["by_queue_gap"]:
            lines.append(
                f"| {labels[method]} | "
                f"{group['queue_gap']} | "
                f"{group['num_states']} | "
                f"{_format_float(group['p_defend_mean'])} | "
                f"{_format_float(group['p_defend_max'])} |"
            )
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def _method_diagnostic(method: str, comparison_row: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(comparison_row["run_dir"])
    summary = _read_json(run_dir / "summary.json")
    inspection = summary["policy_inspection"]
    policy_rows = _read_policy_rows(run_dir / "policy_inspection.jsonl")
    return {
        "method": method,
        "run_dir": str(run_dir),
        "average_cost_mean": float(comparison_row["average_cost_mean"]),
        "num_policy_states": int(inspection["num_policy_states"]),
        "defend_probability_mean": float(inspection["defend_probability_mean"]),
        "defend_probability_max": float(inspection["defend_probability_max"]),
        "defend_probability_threshold": float(
            inspection["defend_probability_threshold"]
        ),
        "num_states_p_defend_at_least_threshold": int(
            inspection["num_states_p_defend_at_least_threshold"]
        ),
        "num_gap_states": int(inspection["num_gap_states"]),
        "num_gap_states_p_defend_at_least_threshold": int(
            inspection["num_gap_states_p_defend_at_least_threshold"]
        ),
        "by_queue_gap": inspection["by_queue_gap"],
        "classification": _classify(inspection),
        "example_rows": _example_rows(policy_rows),
    }


def _classify(inspection: dict[str, Any]) -> str:
    num_states = int(inspection["num_policy_states"])
    defend_states = int(inspection["num_states_p_defend_at_least_threshold"])
    if defend_states == 0:
        return "never_defend"
    if defend_states == num_states:
        return "always_defend"
    return "state_dependent"


def _interpret(aggregate: dict[str, dict[str, Any]]) -> str:
    bvi = aggregate["bvi"]["classification"]
    amq = aggregate["amq"]["classification"]
    nnq = aggregate["nnq"]["classification"]
    if bvi == "state_dependent" and amq == "never_defend" and nnq == "always_defend":
        return (
            "The polling smoke benchmark is runnable but not yet calibrated: BVI "
            "shows a state-dependent defense shape, while AMQ never defends and "
            "NNQ always defends. The smoke cost ranking should not be used as a "
            "performance claim until AMQ/NNQ policy shapes are calibrated."
        )
    return (
        "The polling policy-shape diagnostic does not match the expected BVI "
        "state-dependent / AMQ-NNQ degenerate pattern. Inspect the method rows "
        "before changing budgets."
    )


def _read_policy_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _example_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if int(row["queue_gap"]) > 0:
            examples.append(row)
        if len(examples) == 3:
            break
    return examples


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
