#!/usr/bin/env python3
"""Build the final routing narrative report from existing diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("--cost-loss-diagnostic", required=True)
    parser.add_argument("--bvi-attacker-visitation-diagnostic", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    report = build_narrative(
        comparison_report_path=Path(args.comparison_report),
        cost_loss_diagnostic_path=Path(args.cost_loss_diagnostic),
        bvi_attacker_visitation_diagnostic_path=Path(
            args.bvi_attacker_visitation_diagnostic
        ),
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


def build_narrative(
    *,
    comparison_report_path: Path,
    cost_loss_diagnostic_path: Path,
    bvi_attacker_visitation_diagnostic_path: Path,
) -> dict[str, Any]:
    comparison = _read_json(comparison_report_path)
    cost_loss = _read_json(cost_loss_diagnostic_path)
    visitation = _read_json(bvi_attacker_visitation_diagnostic_path)

    normalized_costs = comparison["runs"]["normalized_amq"][
        "bvi_attacker_average_cost"
    ]
    fitted_costs = comparison["runs"]["fitted_amq"]["bvi_attacker_average_cost"]
    cost_aggregate = cost_loss["aggregate"]
    visitation_aggregate = visitation["aggregate"]
    return {
        "benchmark": "routing",
        "sources": {
            "comparison_report": str(comparison_report_path),
            "cost_loss_diagnostic": str(cost_loss_diagnostic_path),
            "bvi_attacker_visitation_diagnostic": str(
                bvi_attacker_visitation_diagnostic_path
            ),
        },
        "selected_baselines": {
            "main_amq_performance_baseline": "normalized_amq",
            "diagnostic_calibration_baseline": "fitted_amq",
        },
        "performance": {
            "main_metric": comparison["main_metric"],
            "normalized_amq_bvi_attacker_mean": normalized_costs["amq"],
            "fitted_amq_bvi_attacker_mean": fitted_costs["amq"],
            "bvi_bvi_attacker_mean": normalized_costs["bvi"],
            "nnq_bvi_attacker_mean": normalized_costs["nnq"],
            "normalized_amq_margin_vs_nnq": comparison["conclusion"][
                "normalized_amq_margin_vs_nnq"
            ],
            "fitted_amq_delta_vs_normalized": comparison["conclusion"][
                "fitted_amq_cost_delta_vs_normalized"
            ],
        },
        "cost_loss": {
            "bvi_attacker_cost_delta": cost_aggregate[
                "mean_bvi_attacker_cost_delta"
            ],
            "random_attacker_cost_delta": cost_aggregate["mean_average_cost_delta"],
            "always_attack_cost_delta": cost_aggregate[
                "mean_always_attack_cost_delta"
            ],
            "minimax_cost_delta": cost_aggregate["mean_minimax_cost_delta"],
            "policy_gap_delta": cost_aggregate["mean_policy_gap_delta"],
            "defend_probability_delta": cost_aggregate[
                "mean_defend_probability_delta"
            ],
            "num_bvi_attacker_worse_seeds": cost_aggregate["num_cost_worse_seeds"],
        },
        "bvi_attacker_visitation": {
            "visit_weighted_abs_gap_delta": visitation_aggregate[
                "mean_visit_weighted_abs_gap_delta"
            ],
            "visit_weighted_signed_gap_delta": visitation_aggregate[
                "mean_visit_weighted_signed_gap_delta"
            ],
            "visited_amq_defend_probability_delta": visitation_aggregate[
                "mean_visited_p_defend_amq_delta"
            ],
            "visited_bvi_defend_probability_delta": visitation_aggregate[
                "mean_visited_p_defend_bvi_delta"
            ],
            "visited_over_defend_states_delta": visitation_aggregate[
                "mean_visited_over_defend_states_delta"
            ],
        },
        "claim": (
            "Normalized AMQ is the selected routing AMQ performance baseline. "
            "Fitted AMQ is retained as a calibration diagnostic: it improves "
            "policy-shape diagnostics and random/always-attack costs, but it "
            "reduces defense probability on states visited by the common BVI "
            "attacker, which explains its worse BVI-attacker rollout cost."
        ),
        "next_step": (
            "Do not tune fitted calibration blindly. The next algorithmic change "
            "should preserve BVI-attacker-visited defense mass while reducing "
            "unvisited or low-value over-defense."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    performance = report["performance"]
    cost_loss = report["cost_loss"]
    visitation = report["bvi_attacker_visitation"]
    lines = [
        "# Routing Narrative Report",
        "",
        "## Selected Baselines",
        "",
        "- Main AMQ performance baseline: `normalized_amq`",
        "- Diagnostic calibration baseline: `fitted_amq`",
        "",
        "## Performance Result",
        "",
        "Common BVI-attacker average cost, 3 seeds, 50 evaluation episodes:",
        "",
        "| Method | Mean cost |",
        "| --- | ---: |",
        f"| BVI | {_format_float(performance['bvi_bvi_attacker_mean'])} |",
        f"| normalized AMQ | {_format_float(performance['normalized_amq_bvi_attacker_mean'])} |",
        f"| fitted AMQ | {_format_float(performance['fitted_amq_bvi_attacker_mean'])} |",
        f"| NNQ | {_format_float(performance['nnq_bvi_attacker_mean'])} |",
        "",
        (
            "Normalized AMQ beats NNQ by "
            f"{_format_float(performance['normalized_amq_margin_vs_nnq'])}, "
            "but remains behind BVI. Fitted AMQ is worse than normalized AMQ by "
            f"{_format_float(performance['fitted_amq_delta_vs_normalized'])}."
        ),
        "",
        "## Cost-Loss Diagnostic",
        "",
        "Delta is fitted AMQ minus normalized AMQ:",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| BVI-attacker cost | {_format_float(cost_loss['bvi_attacker_cost_delta'])} |",
        f"| random-attacker cost | {_format_float(cost_loss['random_attacker_cost_delta'])} |",
        f"| always-attack cost | {_format_float(cost_loss['always_attack_cost_delta'])} |",
        f"| minimax cost | {_format_float(cost_loss['minimax_cost_delta'])} |",
        f"| policy gap | {_format_float(cost_loss['policy_gap_delta'])} |",
        f"| defend probability | {_format_float(cost_loss['defend_probability_delta'])} |",
        "",
        (
            "BVI-attacker cost worsens on "
            f"{cost_loss['num_bvi_attacker_worse_seeds']} seeds, while random "
            "and always-attack costs improve slightly. The fitted variant is "
            "therefore not globally worse; the loss is concentrated in the "
            "common BVI-attacker evaluation."
        ),
        "",
        "## BVI-Attacker Visitation Diagnostic",
        "",
        "Delta is fitted AMQ minus normalized AMQ on states visited under the common BVI attacker:",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        (
            "| visit-weighted abs policy gap | "
            f"{_format_float(visitation['visit_weighted_abs_gap_delta'])} |"
        ),
        (
            "| visit-weighted signed policy gap | "
            f"{_format_float(visitation['visit_weighted_signed_gap_delta'])} |"
        ),
        (
            "| visited AMQ defend probability | "
            f"{_format_float(visitation['visited_amq_defend_probability_delta'])} |"
        ),
        (
            "| visited BVI defend probability | "
            f"{_format_float(visitation['visited_bvi_defend_probability_delta'])} |"
        ),
        (
            "| visited over-defend states | "
            f"{_format_float(visitation['visited_over_defend_states_delta'])} |"
        ),
        "",
        "## Claim",
        "",
        report["claim"],
        "",
        "## Next Step",
        "",
        report["next_step"],
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
