import json
import subprocess
import sys
import unittest
from pathlib import Path


class ServiceRateComparisonRunnerTests(unittest.TestCase):
    def test_comparison_runner_smoke(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_service_rate_comparison.py"),
                "--config",
                str(root / "configs" / "service_rate_comparison_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("method_summary: method=bvi", completed.stdout)
        self.assertIn("method_summary: method=amq", completed.stdout)
        self.assertIn("method_summary: method=nnq", completed.stdout)
        self.assertIn("method_summary: method=nnq_state0_guard", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "comparison.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["methods"],
            ["bvi", "amq", "nnq", "nnq_state0_guard"],
        )


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"comparison output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
