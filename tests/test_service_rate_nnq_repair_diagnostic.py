import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ServiceRateNNQRepairDiagnosticTests(unittest.TestCase):
    def test_repair_diagnostic_writes_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            config_path = temp_dir / "config.yaml"
            output_json = temp_dir / "repair.json"
            output_md = temp_dir / "repair.md"
            config_path.write_text(_config(temp_dir), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "diagnose_service_rate_nnq_repair.py"),
                    "--config",
                    str(config_path),
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
            self.assertEqual(
                report["algorithm"],
                "nnq_low_state_policy_repair_diagnostic",
            )
            self.assertEqual(report["rows"][0]["label"], "baseline")
            self.assertGreaterEqual(len(report["rows"]), 2)
            self.assertTrue((Path(report["run_dir"]) / "summary.json").exists())
            self.assertIn(
                "Service-Rate NNQ Repair Diagnostic",
                output_md.read_text(encoding="utf-8"),
            )


def _config(output_dir: Path) -> str:
    return f"""
experiment:
  name: nnq_repair_test
  output_dir: {output_dir}

env:
  name: service_rate_control
  gamma: 0.95
  lambda_arrival: 2.0
  mu_levels: [1.0, 3.0, 5.0]
  service_costs: [0.0, 0.5, 2.0]
  attack_cost: 0.5
  q_congestion: 1.0
  initial_state: 0
  uniformization_rate: 7.0
  robust_defender_actions: [2]

algorithm:
  name: nnq

nnq:
  hidden_size: 8
  learning_rate: 0.001
  total_steps: 20
  batch_size: 8
  replay_capacity: 100
  target_update_interval: 10
  epsilon: 0.2
  seed: 0
  log_interval: 10
  state_scale: 10.0

evaluation:
  num_episodes: 2
  horizon: 5
  seed: 200
  tail_threshold: 8
  boundary_state: 20

policy_grid:
  max_state: 2
  high_probability_threshold: 0.5

repair:
  max_repair_state: 1
  defender_actions: [0, 1]
"""


if __name__ == "__main__":
    unittest.main()
