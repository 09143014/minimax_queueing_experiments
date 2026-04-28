import json
import subprocess
import sys
import unittest
from pathlib import Path


class RoutingComparisonRunnerTests(unittest.TestCase):
    def test_routing_comparison_smoke_runs_and_aggregates(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_routing_comparison.py"),
                "--config",
                str(root / "configs" / "routing_comparison_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("bvi_avg_cost=", completed.stdout)
        self.assertIn("amq_avg_cost=", completed.stdout)
        self.assertIn("nnq_avg_cost=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "algorithm_summaries.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "routing")
        self.assertEqual(summary["algorithms"], ["bvi", "amq", "nnq"])
        self.assertEqual(len(summary["rows"]), 3)
        self.assertIn("random_attacker_ranked_by_average_cost", summary["comparison"])
        self.assertIn(
            "random_attacker_average_cost_gap_from_best",
            summary["comparison"],
        )
        self.assertIn("always_attack_ranked_by_average_cost", summary["comparison"])
        self.assertIn(
            "always_attack_average_cost_gap_from_best",
            summary["comparison"],
        )
        self.assertIn("minimax_ranked_by_average_cost", summary["comparison"])
        self.assertIn("minimax_average_cost_gap_from_best", summary["comparison"])
        self.assertIn("bvi_attacker_ranked_by_average_cost", summary["comparison"])
        self.assertIn(
            "bvi_attacker_average_cost_gap_from_best",
            summary["comparison"],
        )
        for row in summary["rows"]:
            self.assertIn("always_attack_average_cost_mean", row)
            self.assertIn("minimax_average_cost_mean", row)
            self.assertIn("bvi_attacker_average_cost_mean", row)
            self.assertTrue(
                (Path(row["run_dir"]) / "evaluation_always_attack.jsonl").exists()
            )
            self.assertTrue((Path(row["run_dir"]) / "evaluation_minimax.jsonl").exists())
            self.assertTrue(
                (Path(row["run_dir"]) / "evaluation_bvi_attacker.jsonl").exists()
            )


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
