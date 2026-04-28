import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RoutingNarrativeReportTests(unittest.TestCase):
    def test_narrative_report_combines_existing_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            comparison = temp_dir / "comparison.json"
            cost_loss = temp_dir / "cost_loss.json"
            visitation = temp_dir / "visitation.json"
            output_json = temp_dir / "narrative.json"
            output_md = temp_dir / "narrative.md"

            comparison.write_text(json.dumps(_comparison_report()), encoding="utf-8")
            cost_loss.write_text(json.dumps(_cost_loss_report()), encoding="utf-8")
            visitation.write_text(json.dumps(_visitation_report()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_routing_narrative_report.py"),
                    "--comparison-report",
                    str(comparison),
                    "--cost-loss-diagnostic",
                    str(cost_loss),
                    "--bvi-attacker-visitation-diagnostic",
                    str(visitation),
                    "--json-output",
                    str(output_json),
                    "--markdown-output",
                    str(output_md),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(
                report["selected_baselines"]["main_amq_performance_baseline"],
                "normalized_amq",
            )
            self.assertAlmostEqual(
                report["performance"]["normalized_amq_bvi_attacker_mean"],
                0.224,
            )
            self.assertAlmostEqual(
                report["cost_loss"]["bvi_attacker_cost_delta"],
                0.002,
            )
            self.assertAlmostEqual(
                report["bvi_attacker_visitation"][
                    "visited_amq_defend_probability_delta"
                ],
                -0.083,
            )
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("Routing Narrative Report", markdown)
            self.assertIn("Selected Baselines", markdown)
            self.assertIn("BVI-Attacker Visitation Diagnostic", markdown)


def _comparison_report() -> dict:
    return {
        "main_metric": "bvi_attacker_average_cost_mean",
        "runs": {
            "normalized_amq": {
                "bvi_attacker_average_cost": {
                    "bvi": 0.218,
                    "amq": 0.224,
                    "nnq": 0.229,
                },
            },
            "fitted_amq": {
                "bvi_attacker_average_cost": {
                    "bvi": 0.218,
                    "amq": 0.226,
                    "nnq": 0.229,
                },
            },
        },
        "conclusion": {
            "normalized_amq_margin_vs_nnq": 0.005,
            "fitted_amq_cost_delta_vs_normalized": 0.002,
        },
    }


def _cost_loss_report() -> dict:
    return {
        "aggregate": {
            "mean_bvi_attacker_cost_delta": 0.002,
            "mean_average_cost_delta": -0.001,
            "mean_always_attack_cost_delta": -0.001,
            "mean_minimax_cost_delta": 0.004,
            "mean_policy_gap_delta": -0.15,
            "mean_defend_probability_delta": -0.18,
            "num_cost_worse_seeds": 3,
        },
    }


def _visitation_report() -> dict:
    return {
        "aggregate": {
            "mean_visit_weighted_abs_gap_delta": -0.074,
            "mean_visit_weighted_signed_gap_delta": -0.092,
            "mean_visited_p_defend_amq_delta": -0.083,
            "mean_visited_p_defend_bvi_delta": 0.009,
            "mean_visited_over_defend_states_delta": -5.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
