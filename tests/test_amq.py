import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.amq import AMQConfig, LinearAMQTrainer
from adversarial_queueing.envs.routing import RoutingConfig, RoutingEnv
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

    def test_routing_training_smoke_logs_tuple_states(self):
        env = RoutingEnv(
            RoutingConfig(
                lambda_arrival=2.0,
                mu_rates=(1.0, 1.5, 2.0),
                gamma=0.95,
                uniformization_rate=6.5,
                bvi_max_queue_length=3,
            )
        )
        trainer = LinearAMQTrainer(
            env,
            AMQConfig(
                feature_set="basic",
                total_steps=25,
                eta0=0.001,
                seed=123,
                log_interval=5,
            ),
        )

        result = trainer.train()

        self.assertEqual(result.metrics[-1]["step"], 25)
        self.assertTrue(np.isfinite(result.weights).all())
        self.assertGreater(np.linalg.norm(result.weights), 0.0)
        self.assertIsInstance(result.final_state, tuple)
        self.assertIsInstance(result.metrics[-1]["state"], list)

    def test_routing_exploring_starts_are_bounded(self):
        env = RoutingEnv(
            RoutingConfig(
                lambda_arrival=2.0,
                mu_rates=(1.0, 1.5, 2.0),
                gamma=0.95,
                uniformization_rate=6.5,
                bvi_max_queue_length=3,
            )
        )
        trainer = LinearAMQTrainer(
            env,
            AMQConfig(
                feature_set="basic",
                total_steps=10,
                eta0=0.001,
                seed=123,
                exploring_starts_probability=1.0,
                exploring_starts_max_queue_length=3,
            ),
        )

        result = trainer.train()

        logged_state = result.metrics[0]["state"]
        self.assertTrue(all(0 <= value <= 3 for value in logged_state))


if __name__ == "__main__":
    unittest.main()
