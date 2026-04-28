import json
import subprocess
import sys
import unittest
from pathlib import Path


class ServiceRateComparisonMultiseedRunnerTests(unittest.TestCase):
    def test_service_rate_comparison_multiseed_smoke(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_service_rate_comparison_multiseed.py"),
                "--config",
                str(root / "configs" / "service_rate_comparison_multiseed_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("best_counts=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "seed_method_summaries.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "service_rate_control")
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(len(summary["rows"]), 2)
        self.assertIn("average_cost", summary["ranking_counts"])
        for method in ("bvi", "amq", "nnq", "nnq_state0_guard"):
            self.assertIn(method, summary["aggregate"])
            self.assertIn("average_cost_mean", summary["aggregate"][method])


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
