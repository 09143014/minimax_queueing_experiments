import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ServiceRatePolicyShapeDiagnosticTests(unittest.TestCase):
    def test_policy_shape_diagnostic_detects_empty_state_over_service(self):
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
                (run_dir / "policy_grid.jsonl").write_text(
                    json.dumps(_grid_row(method)) + "\n",
                    encoding="utf-8",
                )
            summary.write_text(json.dumps(_summary(run_dirs)), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "diagnose_service_rate_policy_shape.py"),
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
            self.assertAlmostEqual(
                report["aggregate"]["nnq"]["mean_p_high_state_0"],
                1.0,
            )
            self.assertIn("over-service", report["interpretation"])
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("Service-Rate Policy-Shape Diagnostic", markdown)


def _grid_row(method: str) -> dict:
    if method == "nnq":
        return {"method": method, "state": 0, "p_low": 0.0, "p_medium": 0.0, "p_high": 1.0}
    if method == "amq":
        return {"method": method, "state": 0, "p_low": 0.0, "p_medium": 1.0, "p_high": 0.0}
    return {"method": method, "state": 0, "p_low": 1.0, "p_medium": 0.0, "p_high": 0.0}


def _summary(run_dirs: dict[str, Path]) -> dict:
    return {
        "seeds": [0],
        "rows": [
            {
                "seed": 0,
                "method_rows": [
                    _method_row("bvi", run_dirs["bvi"], 0.3, 1),
                    _method_row("amq", run_dirs["amq"], 0.35, 1),
                    _method_row("nnq", run_dirs["nnq"], 0.5, 0),
                ],
            },
        ],
    }


def _method_row(method: str, run_dir: Path, cost: float, first_high: int) -> dict:
    return {
        "method": method,
        "run_dir": str(run_dir),
        "average_cost_mean": cost,
        "first_state_p_high_at_least_threshold": first_high,
    }


if __name__ == "__main__":
    unittest.main()
