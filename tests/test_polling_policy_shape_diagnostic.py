import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PollingPolicyShapeDiagnosticTests(unittest.TestCase):
    def test_polling_policy_shape_diagnostic_detects_degenerate_shapes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            summary = temp_dir / "summary.json"
            output_json = temp_dir / "diagnostic.json"
            output_md = temp_dir / "diagnostic.md"
            run_dirs = {}
            for method in ("bvi", "amq", "nnq"):
                run_dir = temp_dir / method
                run_dir.mkdir()
                run_dirs[method] = run_dir
                inspection = _inspection(method)
                (run_dir / "summary.json").write_text(
                    json.dumps({"policy_inspection": inspection}),
                    encoding="utf-8",
                )
                (run_dir / "policy_inspection.jsonl").write_text(
                    json.dumps(_policy_row(method)) + "\n",
                    encoding="utf-8",
                )
            summary.write_text(json.dumps(_summary(run_dirs)), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "diagnose_polling_policy_shape.py"),
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
            self.assertEqual(report["benchmark"], "polling")
            self.assertEqual(report["aggregate"]["amq"]["classification"], "never_defend")
            self.assertEqual(report["aggregate"]["nnq"]["classification"], "always_defend")
            self.assertIn("not yet calibrated", report["interpretation"])
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("Polling Policy-Shape Diagnostic", markdown)


def _summary(run_dirs: dict[str, Path]) -> dict:
    return {
        "benchmark": "polling",
        "rows": [
            _method_row("bvi", run_dirs["bvi"], 0.4),
            _method_row("amq", run_dirs["amq"], 0.5),
            _method_row("nnq", run_dirs["nnq"], 0.3),
        ],
    }


def _method_row(method: str, run_dir: Path, cost: float) -> dict:
    return {
        "method": method,
        "run_dir": str(run_dir),
        "average_cost_mean": cost,
    }


def _inspection(method: str) -> dict:
    if method == "bvi":
        defend = 1
        mean = 0.5
    elif method == "amq":
        defend = 0
        mean = 0.0
    else:
        defend = 2
        mean = 1.0
    return {
        "num_policy_states": 2,
        "defend_probability_mean": mean,
        "defend_probability_max": mean,
        "defend_probability_threshold": 0.5,
        "num_states_p_defend_at_least_threshold": defend,
        "num_gap_states": 1,
        "num_gap_states_p_defend_at_least_threshold": min(defend, 1),
        "by_queue_gap": [
            {
                "queue_gap": 0,
                "num_states": 1,
                "p_defend_mean": mean,
                "p_defend_max": mean,
            }
        ],
    }


def _policy_row(method: str) -> dict:
    p_defend = {"bvi": 0.5, "amq": 0.0, "nnq": 1.0}[method]
    return {
        "method": method,
        "state": [0, 1, 0],
        "queue_gap": 1,
        "p_defend": p_defend,
    }


if __name__ == "__main__":
    unittest.main()
