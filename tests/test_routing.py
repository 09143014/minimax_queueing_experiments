import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.envs.routing import RoutingConfig, RoutingEnv
from adversarial_queueing.utils.config import build_routing_config, load_config


class RoutingTests(unittest.TestCase):
    def make_env(self):
        config = RoutingConfig(
            lambda_arrival=2.0,
            mu_rates=(1.0, 1.5, 2.0),
            gamma=0.95,
            uniformization_rate=7.0,
            bvi_max_queue_length=5,
        )
        return RoutingEnv(config)

    def test_attack_routes_to_longest_without_defense(self):
        env = self.make_env()

        self.assertEqual(
            env.routed_arrival_targets((1, 4, 2), attacker_action=1, defender_action=0),
            (1,),
        )

    def test_nominal_or_defended_arrival_routes_to_shortest(self):
        env = self.make_env()

        self.assertEqual(
            env.routed_arrival_targets((1, 4, 2), attacker_action=0, defender_action=0),
            (0,),
        )
        self.assertEqual(
            env.routed_arrival_targets((1, 4, 2), attacker_action=1, defender_action=1),
            (0,),
        )

    def test_tied_targets_split_arrival_probability(self):
        env = self.make_env()

        probs = env.transition_probabilities(
            (1, 1, 3), attacker_action=0, defender_action=0
        )

        self.assertAlmostEqual(probs[(2, 1, 3)], 1.0 / 7.0)
        self.assertAlmostEqual(probs[(1, 2, 3)], 1.0 / 7.0)

    def test_transition_probabilities_sum_to_one(self):
        env = self.make_env()

        probs = env.transition_probabilities(
            (1, 2, 0), attacker_action=1, defender_action=0
        )

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertTrue(all(prob >= 0.0 for prob in probs.values()))
        self.assertNotIn((1, 2, -1), probs)

    def test_boundary_clips_arrivals(self):
        env = self.make_env()

        probs = env.transition_probabilities(
            (5, 1, 2), attacker_action=1, defender_action=0
        )

        self.assertIn((5, 1, 2), probs)
        self.assertNotIn((6, 1, 2), probs)

    def test_routing_smoke_config_builds(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "routing_smoke.yaml"
        config = build_routing_config(load_config(path))

        self.assertEqual(config.mu_rates, (1.0, 1.5, 2.0))
        self.assertEqual(config.initial_state_value, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
