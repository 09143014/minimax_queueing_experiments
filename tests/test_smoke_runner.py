import json
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
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "evaluation.jsonl").exists())
        self.assertTrue((run_dir / "policy_grid.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("evaluation", summary)
        self.assertIn("average_cost_mean", summary["evaluation"])
        self.assertIn("policy_grid", summary)
        self.assertIn("first_state_p_high_at_least_threshold", summary["policy_grid"])

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
        self.assertTrue((run_dir / "policy_grid.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("evaluation", summary)
        self.assertIn("average_cost_mean", summary["evaluation"])
        self.assertIn("policy_grid", summary)
        self.assertIn("first_state_p_high_at_least_threshold", summary["policy_grid"])

    def test_bvi_sensitivity_config_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_experiment.py"),
                "--config",
                str(root / "configs" / "bvi_sensitivity.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("threshold_stable=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "sensitivity.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("sensitivity", summary)
        self.assertIn("value_range_at_eval_state", summary["sensitivity"])

    def test_routing_bvi_smoke_config_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_experiment.py"),
                "--config",
                str(root / "configs" / "routing_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("states=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "evaluation.jsonl").exists())
        self.assertTrue((run_dir / "policy_inspection.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark"], "routing")
        self.assertEqual(summary["num_states"], 64)
        self.assertIn("value_at_initial_state", summary)
        self.assertIn("evaluation", summary)
        self.assertIn("policy_inspection", summary)
        self.assertIn(
            "num_states_p_defend_at_least_threshold",
            summary["policy_inspection"],
        )

    def test_nnq_smoke_config_runs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_experiment.py"),
                "--config",
                str(root / "configs" / "nnq_smoke.yaml"),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("final_loss=", completed.stdout)
        run_dir = _run_dir_from_stdout(completed.stdout)
        self.assertTrue((run_dir / "metrics.jsonl").exists())
        self.assertTrue((run_dir / "evaluation.jsonl").exists())
        self.assertTrue((run_dir / "policy_grid.jsonl").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["implementation"], "numpy_mlp_smoke")
        self.assertIn("evaluation", summary)
        self.assertIn("policy_grid", summary)


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise AssertionError(f"runner output did not include run directory: {stdout}")


if __name__ == "__main__":
    unittest.main()
