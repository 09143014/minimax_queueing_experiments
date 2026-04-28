import json
import subprocess
import sys
import unittest
from pathlib import Path


class RoutingComparisonMultiseedRunnerTests(unittest.TestCase):
    def test_routing_comparison_multiseed_smoke_runs_and_aggregates(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_routing_comparison_multiseed.py"),
                "--config",
                str(root / "configs" / "routing_comparison_multiseed_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("bvi_attacker_best_counts=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "seed_algorithm_summaries.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "routing")
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(len(summary["rows"]), 2)
        self.assertIn("bvi_attacker", summary["ranking_counts"])
        for algorithm in ("bvi", "amq", "nnq"):
            self.assertIn(algorithm, summary["aggregate"])
            self.assertIn(
                "bvi_attacker_average_cost_mean",
                summary["aggregate"][algorithm],
            )


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
