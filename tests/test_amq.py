import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.amq import AMQConfig, LinearAMQTrainer
from adversarial_queueing.envs.service_rate_control import (
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)


class AMQTests(unittest.TestCase):
    def make_env(self):
        config = ServiceRateControlConfig(
            lambda_arrival=2.0,
            mu_levels=(1.0, 3.0, 5.0),
            service_costs=(0.0, 0.5, 2.0),
            gamma=0.95,
            uniformization_rate=7.0,
            bvi_max_queue_length=10,
        )
        return ServiceRateControlEnv(config)

    def test_training_smoke_logs_metrics_and_updates_weights(self):
        trainer = LinearAMQTrainer(
            self.make_env(),
            AMQConfig(total_steps=25, eta0=0.001, seed=123, log_interval=5),
        )

        result = trainer.train()

        self.assertEqual(result.metrics[-1]["step"], 25)
        self.assertTrue(np.isfinite(result.weights).all())
        self.assertGreater(np.linalg.norm(result.weights), 0.0)
        self.assertIn("td_error", result.metrics[-1])
        self.assertIn("minimax_value_next", result.metrics[-1])


if __name__ == "__main__":
    unittest.main()

