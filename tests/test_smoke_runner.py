import subprocess
import sys
import unittest
import json
from pathlib import Path


class SmokeRunnerTests(unittest.TestCase):
    def test_smoke_config_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_experiment.py"),
                "--config",
                str(root / "configs" / "smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("summary:", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "evaluation.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("evaluation", summary)
        self.assertIn("average_cost_mean", summary["evaluation"])

    def test_amq_smoke_config_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_experiment.py"),
                "--config",
                str(root / "configs" / "amq_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("final_td_error=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "evaluation.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("evaluation", summary)
        self.assertIn("average_cost_mean", summary["evaluation"])


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
