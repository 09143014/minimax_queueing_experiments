import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.algorithms.amq import AMQConfig, LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration
from adversarial_queueing.algorithms.nnq import NNQConfig, NNQTrainer
from adversarial_queueing.envs.polling import PollingConfig, PollingEnv
from adversarial_queueing.evaluation.polling_policy import (
    amq_polling_policy_inspection,
    bvi_polling_policy_inspection,
    nnq_polling_policy_inspection,
)


class PollingPolicyInspectionTests(unittest.TestCase):
    def make_env(self):
        return PollingEnv(
            PollingConfig(
                lambda_arrivals=(1.0, 1.0),
                mu_service=1.5,
                gamma=0.9,
                uniformization_rate=4.0,
                bvi_max_queue_length=2,
            )
        )

    def test_bvi_policy_inspection_exports_rows_and_gap_summary(self):
        env = self.make_env()
        states = [(x0, x1, p) for x0 in range(3) for x1 in range(3) for p in range(2)]
        result = run_bounded_value_iteration(
            env,
            max_queue_length=2,
            tolerance=1e-5,
            max_iterations=500,
            states=states,
        )

        rows, summary = bvi_polling_policy_inspection(env, result)

        self.assertEqual(len(rows), len(states))
        self.assertIn("p_defend", rows[0])
        self.assertIn("queue_gap", rows[0])
        self.assertIn("nominal_targets", rows[0])
        self.assertIn("attacked_targets", rows[0])
        self.assertEqual(summary["num_policy_states"], len(states))
        self.assertIn("num_gap_states_p_defend_at_least_threshold", summary)
        self.assertIn("by_queue_gap", summary)

    def test_amq_policy_inspection_exports_bounded_grid(self):
        env = self.make_env()
        trainer = LinearAMQTrainer(
            env,
            AMQConfig(feature_set="basic", total_steps=10, eta0=0.001, seed=7),
        )
        trainer.train()

        rows, summary = amq_polling_policy_inspection(env, trainer, max_queue_length=2)

        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0]["method"], "amq")
        self.assertEqual(summary["num_policy_states"], 18)

    def test_nnq_policy_inspection_exports_bounded_grid(self):
        env = self.make_env()
        trainer = NNQTrainer(
            env,
            NNQConfig(total_steps=20, batch_size=8, hidden_size=8, seed=7),
        )
        trained = trainer.train()
        trainer.network = trained.network
        trainer.target_network = trained.network.copy()

        rows, summary = nnq_polling_policy_inspection(env, trainer, max_queue_length=2)

        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0]["method"], "nnq")
        self.assertEqual(summary["num_policy_states"], 18)


if __name__ == "__main__":
    unittest.main()
