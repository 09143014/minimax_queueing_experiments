import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.envs.service_rate_control import (
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)
from adversarial_queueing.evaluation.rollout import (
    EvaluationConfig,
    evaluate_policy,
    random_attacker_policy,
)


class RolloutEvaluationTests(unittest.TestCase):
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

    def test_fixed_seed_rollout_is_deterministic(self):
        config = EvaluationConfig(num_episodes=2, horizon=5, seed=50, tail_threshold=8)

        first = evaluate_policy(
            self.make_env(),
            defender_policy=lambda state, rng, env: 2,
            attacker_policy=random_attacker_policy,
            config=config,
        )
        second = evaluate_policy(
            self.make_env(),
            defender_policy=lambda state, rng, env: 2,
            attacker_policy=random_attacker_policy,
            config=config,
        )

        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.summary, second.summary)
        self.assertIn("average_cost_mean", first.summary)
        self.assertIn("discounted_cost_mean", first.summary)
        self.assertIn("boundary_hit_fraction_mean", first.summary)


if __name__ == "__main__":
    unittest.main()

