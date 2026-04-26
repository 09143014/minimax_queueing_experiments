import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.bvi import (
    bounded_queue_states,
    run_bounded_value_iteration,
)
from adversarial_queueing.envs.routing import RoutingConfig, RoutingEnv


class BVITests(unittest.TestCase):
    def test_bounded_queue_states_for_routing(self):
        states = bounded_queue_states(num_queues=2, max_queue_length=2)

        self.assertEqual(len(states), 9)
        self.assertIn((0, 0), states)
        self.assertIn((2, 2), states)

    def test_routing_bvi_runs_on_explicit_state_space(self):
        config = RoutingConfig(
            lambda_arrival=1.0,
            mu_rates=(1.0, 1.0),
            gamma=0.9,
            uniformization_rate=3.0,
            bvi_max_queue_length=2,
        )
        env = RoutingEnv(config)

        result = run_bounded_value_iteration(
            env,
            max_queue_length=2,
            tolerance=1e-5,
            max_iterations=500,
            states=bounded_queue_states(num_queues=2, max_queue_length=2),
        )

        self.assertEqual(len(result.values), 9)
        self.assertIn((0, 0), result.values)
        self.assertLess(result.residual, 1e-5)


if __name__ == "__main__":
    unittest.main()
