import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.amq import AMQConfig, LinearAMQTrainer
from adversarial_queueing.envs.service_rate_control import (
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)
from adversarial_queueing.evaluation.policy_grid import PolicyGridConfig, amq_policy_grid


class PolicyGridTests(unittest.TestCase):
    def test_amq_policy_grid_schema_and_probabilities(self):
        env = ServiceRateControlEnv(
            ServiceRateControlConfig(
                lambda_arrival=2.0,
                mu_levels=(1.0, 3.0, 5.0),
                service_costs=(0.0, 0.5, 2.0),
                gamma=0.95,
                uniformization_rate=7.0,
                bvi_max_queue_length=5,
            )
        )
        trainer = LinearAMQTrainer(env, AMQConfig(total_steps=5, seed=0))
        trainer.train()

        rows, summary = amq_policy_grid(env, trainer, PolicyGridConfig(max_state=4))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["method"], "amq")
        self.assertEqual(rows[-1]["state"], 4)
        for row in rows:
            total = row["p_low"] + row["p_medium"] + row["p_high"]
            self.assertAlmostEqual(total, 1.0)
        self.assertEqual(summary["policy_grid_max_state"], 4)
        self.assertIn("first_state_p_high_at_least_threshold", summary)


if __name__ == "__main__":
    unittest.main()

