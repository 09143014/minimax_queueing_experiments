import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.nnq import NNQConfig, NNQTrainer
from adversarial_queueing.envs.service_rate_control import (
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)


class NNQTests(unittest.TestCase):
    def make_env(self):
        return ServiceRateControlEnv(
            ServiceRateControlConfig(
                lambda_arrival=2.0,
                mu_levels=(1.0, 3.0, 5.0),
                service_costs=(0.0, 0.5, 2.0),
                gamma=0.95,
                uniformization_rate=7.0,
                bvi_max_queue_length=10,
            )
        )

    def test_training_smoke_logs_metrics_and_outputs_matrix(self):
        trainer = NNQTrainer(
            self.make_env(),
            NNQConfig(total_steps=30, batch_size=8, hidden_size=8, seed=123, log_interval=10),
        )

        result = trainer.train()
        trainer.network = result.network

        self.assertEqual(result.metrics[-1]["step"], 30)
        self.assertTrue(np.isfinite(trainer.q_matrix(0)).all())
        self.assertEqual(trainer.q_matrix(0).shape, (2, 3))
        self.assertIn("loss", result.metrics[-1])


if __name__ == "__main__":
    unittest.main()

