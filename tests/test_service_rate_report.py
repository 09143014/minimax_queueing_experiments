import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ServiceRateReportTests(unittest.TestCase):
    def test_service_rate_report_builder(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            summary = temp_dir / "summary.json"
            output_json = temp_dir / "report.json"
            output_md = temp_dir / "report.md"
            summary.write_text(json.dumps(_summary()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_service_rate_report.py"),
                    "--summary",
                    str(summary),
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
            self.assertEqual(report["benchmark"], "service_rate_control")
            self.assertAlmostEqual(report["conclusion"]["amq_margin_vs_nnq"], 0.15)
            self.assertAlmostEqual(report["conclusion"]["amq_gap_vs_bvi"], 0.05)
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("Service-Rate-Control Report", markdown)
            self.assertIn("Per-Seed Average Cost", markdown)


def _summary() -> dict:
    return {
        "num_seeds": 1,
        "seeds": [0],
        "aggregate": {
            "bvi": {
                "average_cost_mean": {"mean": 0.30},
                "first_state_p_high_at_least_threshold": {"mean": 1.0},
            },
            "amq": {
                "average_cost_mean": {"mean": 0.35},
                "first_state_p_high_at_least_threshold": {"mean": 1.0},
            },
            "nnq": {
                "average_cost_mean": {"mean": 0.50},
                "first_state_p_high_at_least_threshold": {"mean": 0.0},
            },
        },
        "ranking_counts": {
            "average_cost": {
                "bvi": 1,
                "amq": 0,
                "nnq": 0,
            },
        },
        "rows": [
            {
                "seed": 0,
                "method_rows": [
                    {"method": "bvi", "average_cost_mean": 0.30},
                    {"method": "amq", "average_cost_mean": 0.35},
                    {"method": "nnq", "average_cost_mean": 0.50},
                ],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
