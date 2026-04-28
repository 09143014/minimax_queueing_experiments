import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RoutingComparisonReportTests(unittest.TestCase):
    def test_report_builder_writes_json_and_markdown(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            normalized_summary = temp_dir / "normalized_summary.json"
            fitted_summary = temp_dir / "fitted_summary.json"
            normalized_visitation = temp_dir / "normalized_visitation.json"
            fitted_visitation = temp_dir / "fitted_visitation.json"
            report_json = temp_dir / "report.json"
            report_md = temp_dir / "report.md"

            normalized_summary.write_text(
                json.dumps(_comparison_summary(amq_mean=0.224, amq_best_count=1)),
                encoding="utf-8",
            )
            fitted_summary.write_text(
                json.dumps(_comparison_summary(amq_mean=0.226, amq_best_count=0)),
                encoding="utf-8",
            )
            normalized_visitation.write_text(
                json.dumps(_visitation_summary(over_defend=26, gap=0.597)),
                encoding="utf-8",
            )
            fitted_visitation.write_text(
                json.dumps(_visitation_summary(over_defend=13, gap=0.398)),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_routing_comparison_report.py"),
                    "--normalized-summary",
                    str(normalized_summary),
                    "--fitted-summary",
                    str(fitted_summary),
                    "--normalized-visitation-summary",
                    str(normalized_visitation),
                    "--fitted-visitation-summary",
                    str(fitted_visitation),
                    "--json-output",
                    str(report_json),
                    "--markdown-output",
                    str(report_md),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["benchmark"], "routing")
            self.assertEqual(
                report["conclusion"]["main_amq_performance_baseline"],
                "normalized_amq",
            )
            self.assertAlmostEqual(
                report["conclusion"]["fitted_amq_cost_delta_vs_normalized"],
                0.002,
            )
            self.assertEqual(
                report["visitation"]["fitted_amq"][
                    "num_visited_over_defend_states"
                ],
                13,
            )
            markdown = report_md.read_text(encoding="utf-8")
            self.assertIn("Common BVI-attacker average cost", markdown)
            self.assertIn("visited over-defend states", markdown)


def _comparison_summary(amq_mean: float, amq_best_count: int) -> dict:
    return {
        "aggregate": {
            "bvi": {
                "bvi_attacker_average_cost_mean": {
                    "mean": 0.218,
                },
            },
            "amq": {
                "bvi_attacker_average_cost_mean": {
                    "mean": amq_mean,
                },
            },
            "nnq": {
                "bvi_attacker_average_cost_mean": {
                    "mean": 0.229,
                },
            },
        },
        "ranking_counts": {
            "bvi_attacker": {
                "bvi": 3 - amq_best_count,
                "amq": amq_best_count,
                "nnq": 0,
            },
        },
        "rows": [
            {
                "seed": 0,
                "algorithm_rows": [
                    _algorithm_row("bvi", 0.218),
                    _algorithm_row("amq", amq_mean),
                    _algorithm_row("nnq", 0.229),
                ],
            },
        ],
    }


def _algorithm_row(algorithm: str, cost: float) -> dict:
    return {
        "algorithm": algorithm,
        "bvi_attacker_average_cost_mean": cost,
    }


def _visitation_summary(over_defend: int, gap: float) -> dict:
    return {
        "num_states": 64,
        "num_visited_states": 40,
        "num_visited_over_defend_states": over_defend,
        "visit_weighted_abs_gap_mean": gap,
        "visited_abs_gap_mean": gap - 0.01,
        "unvisited_abs_gap_mean": gap + 0.01,
    }


if __name__ == "__main__":
    unittest.main()
