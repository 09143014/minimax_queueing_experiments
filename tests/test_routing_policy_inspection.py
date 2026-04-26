import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.bvi import (
    bounded_queue_states,
    run_bounded_value_iteration,
)
from adversarial_queueing.algorithms.amq import AMQConfig, LinearAMQTrainer
from adversarial_queueing.envs.routing import RoutingConfig, RoutingEnv
from adversarial_queueing.evaluation.routing_policy import bvi_routing_policy_inspection
from adversarial_queueing.evaluation.routing_policy import amq_routing_policy_inspection
from adversarial_queueing.evaluation.rollout import (
    EvaluationConfig,
    evaluate_policy,
    make_bvi_defender_policy,
    random_attacker_policy,
)


class RoutingPolicyInspectionTests(unittest.TestCase):
    def make_result(self):
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
        return env, result

    def test_policy_inspection_exports_rows_and_summary(self):
        env, result = self.make_result()

        rows, summary = bvi_routing_policy_inspection(env, result)

        self.assertEqual(len(rows), len(result.values))
        self.assertIn("p_defend", rows[0])
        self.assertIn("nominal_targets", rows[0])
        self.assertEqual(summary["num_policy_states"], len(result.values))
        self.assertIn("num_states_p_defend_at_least_threshold", summary)

    def test_amq_policy_inspection_exports_bounded_grid(self):
        env, _result = self.make_result()
        trainer = LinearAMQTrainer(
            env,
            AMQConfig(feature_set="basic", total_steps=10, eta0=0.001, seed=7),
        )
        trainer.train()

        rows, summary = amq_routing_policy_inspection(env, trainer, max_queue_length=2)

        self.assertEqual(len(rows), 9)
        self.assertEqual(summary["num_policy_states"], 9)
        self.assertEqual(rows[0]["method"], "amq")

    def test_rollout_summarizes_tuple_states_by_load(self):
        env, result = self.make_result()

        rollout = evaluate_policy(
            env,
            defender_policy=make_bvi_defender_policy(result),
            attacker_policy=random_attacker_policy,
            config=EvaluationConfig(num_episodes=2, horizon=5, seed=10),
        )

        self.assertIn("final_load", rollout.rows[0])
        self.assertIn("final_load_mean", rollout.summary)


if __name__ == "__main__":
    unittest.main()
