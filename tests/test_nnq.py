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
from adversarial_queueing.envs.routing import RoutingConfig, RoutingEnv


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

    def test_routing_augmented_features_expand_state_representation(self):
        env = RoutingEnv(
            RoutingConfig(
                lambda_arrival=2.0,
                mu_rates=(1.0, 1.5, 2.0),
                gamma=0.95,
                uniformization_rate=6.5,
            )
        )
        trainer = NNQTrainer(
            env,
            NNQConfig(
                total_steps=1,
                hidden_size=8,
                state_feature_set="routing_augmented",
                seed=123,
            ),
        )

        self.assertEqual(trainer.network.input_size, 19)
        self.assertEqual(trainer.q_matrix((0, 1, 2)).shape, (2, 2))

    def test_forced_defender_action_overrides_behavior_policy(self):
        trainer = NNQTrainer(
            self.make_env(),
            NNQConfig(
                total_steps=1,
                hidden_size=8,
                epsilon=0.0,
                forced_defender_action_probability=1.0,
                forced_defender_action=2,
                seed=123,
            ),
        )

        _attacker_action, defender_action = trainer._behavior_actions(0)

        self.assertEqual(defender_action, 2)


if __name__ == "__main__":
    unittest.main()
