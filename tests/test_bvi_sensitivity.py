import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.envs.service_rate_control import ServiceRateControlConfig
from adversarial_queueing.evaluation.bvi_sensitivity import (
    BVISensitivityConfig,
    run_bvi_sensitivity,
)
from adversarial_queueing.evaluation.policy_grid import PolicyGridConfig
from adversarial_queueing.evaluation.rollout import EvaluationConfig


class BVISensitivityTests(unittest.TestCase):
    def test_sensitivity_rows_and_summary_schema(self):
        rows, summary = run_bvi_sensitivity(
            ServiceRateControlConfig(
                lambda_arrival=2.0,
                mu_levels=(1.0, 3.0, 5.0),
                service_costs=(0.0, 0.5, 2.0),
                gamma=0.95,
                uniformization_rate=7.0,
                bvi_max_queue_length=5,
            ),
            BVISensitivityConfig(max_queue_lengths=(3, 5), max_iterations=500),
            EvaluationConfig(num_episodes=2, horizon=5, seed=10, boundary_state=5),
            PolicyGridConfig(max_state=5),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["bounds"], [3, 5])
        self.assertIn("value_range_at_eval_state", summary)
        self.assertIn("policy_high_threshold_stable", summary)
        self.assertIn("boundary_hit_fraction_mean", rows[0])


if __name__ == "__main__":
    unittest.main()

