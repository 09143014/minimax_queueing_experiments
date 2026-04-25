import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.envs.service_rate_control import (
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)


class ServiceRateControlTests(unittest.TestCase):
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

    def test_attack_forces_low_rate_without_robust_defense(self):
        env = self.make_env()

        self.assertEqual(env.realized_mu(attacker_action=1, defender_action=0), 1.0)
        self.assertEqual(env.realized_mu(attacker_action=1, defender_action=1), 1.0)
        self.assertEqual(env.realized_mu(attacker_action=1, defender_action=2), 5.0)

    def test_transition_probabilities_sum_to_one(self):
        env = self.make_env()

        probs = env.transition_probabilities(state=3, attacker_action=1, defender_action=2)

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertTrue(all(prob >= 0.0 for prob in probs.values()))

    def test_empty_queue_has_no_negative_service_transition(self):
        env = self.make_env()

        probs = env.transition_probabilities(state=0, attacker_action=0, defender_action=2)

        self.assertNotIn(-1, probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_quadratic_congestion_cost(self):
        env = self.make_env()

        low = env.instantaneous_cost(state=2, attacker_action=0, defender_action=0)
        high = env.instantaneous_cost(state=3, attacker_action=0, defender_action=0)

        self.assertEqual(low, 4.0)
        self.assertEqual(high, 9.0)


if __name__ == "__main__":
    unittest.main()
