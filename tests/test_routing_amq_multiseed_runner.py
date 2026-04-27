import json
import subprocess
import sys
import unittest
from pathlib import Path


class RoutingAMQMultiseedRunnerTests(unittest.TestCase):
    def test_multiseed_smoke_runs_and_aggregates(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_routing_amq_multiseed.py"),
                "--config",
                str(root / "configs" / "routing_amq_multiseed_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("summary: seeds=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "seed_summaries.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "routing")
        self.assertEqual(summary["algorithm"], "amq")
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertIn("policy_gap_mean", summary["aggregate"])
        self.assertIn("q_reference_gap_mean", summary["aggregate"])


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
