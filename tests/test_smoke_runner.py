import subprocess
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()

