import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.envs.polling import PollingConfig, PollingEnv
from adversarial_queueing.utils.config import build_polling_config, load_config


class PollingTests(unittest.TestCase):
    def make_env(self):
        return PollingEnv(
            PollingConfig(
                lambda_arrivals=(1.0, 1.5),
                mu_service=2.0,
                gamma=0.95,
                attack_cost=0.5,
                defend_cost=0.2,
                switch_cost=0.3,
                initial_queues=(1, 3),
                initial_position=0,
                uniformization_rate=5.0,
                bvi_max_queue_length=4,
            )
        )

    def test_nominal_or_defended_polls_longest_queue(self):
        env = self.make_env()

        self.assertEqual(env.polling_targets((1, 3, 0), 0, 0), (1,))
        self.assertEqual(env.polling_targets((1, 3, 0), 1, 1), (1,))

    def test_successful_attack_polls_shortest_queue(self):
        env = self.make_env()

        self.assertEqual(env.polling_targets((1, 3, 0), 1, 0), (0,))

    def test_switching_cost_is_charged_in_expectation(self):
        env = self.make_env()

        stay = env.instantaneous_cost((1, 3, 0), 1, 0)
        switch = env.instantaneous_cost((1, 3, 0), 0, 0)

        self.assertAlmostEqual(switch - stay, 0.3 + 0.5)

    def test_transition_probabilities_sum_to_one_and_update_position(self):
        env = self.make_env()

        probs = env.transition_probabilities((1, 3, 0), 0, 0)

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertTrue(all(prob >= 0.0 for prob in probs.values()))
        self.assertTrue(all(state[-1] == 1 for state in probs))

    def test_empty_selected_queue_service_does_not_go_negative(self):
        env = self.make_env()

        probs = env.transition_probabilities((0, 3, 1), 1, 0)

        self.assertNotIn((-1, 3, 0), probs)
        self.assertIn((0, 3, 0), probs)

    def test_polling_smoke_config_builds(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "polling_smoke.yaml"
        config = build_polling_config(load_config(path))

        self.assertEqual(config.lambda_arrivals, (1.0, 1.5))
        self.assertEqual(config.initial_state_value, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
