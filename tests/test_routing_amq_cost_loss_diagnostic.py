import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RoutingAMQCostLossDiagnosticTests(unittest.TestCase):
    def test_cost_loss_diagnostic_writes_json_and_markdown(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir_string:
            temp_dir = Path(temp_dir_string)
            normalized_child = temp_dir / "normalized_child"
            fitted_child = temp_dir / "fitted_child"
            normalized_child.mkdir()
            fitted_child.mkdir()
            (normalized_child / "summary.json").write_text(
                json.dumps(_child_summary(policy_gap=0.8, q_gap=1.2)),
                encoding="utf-8",
            )
            (fitted_child / "summary.json").write_text(
                json.dumps(_child_summary(policy_gap=0.3, q_gap=1.5)),
                encoding="utf-8",
            )
            normalized_summary = temp_dir / "normalized_summary.json"
            fitted_summary = temp_dir / "fitted_summary.json"
            normalized_summary.write_text(
                json.dumps(
                    _multiseed_summary(
                        run_dir=normalized_child,
                        bvi_attacker_cost=0.224,
                        policy_gap=0.8,
                        q_gap=1.2,
                        defend_probability=0.9,
                        defend_states=60,
                    )
                ),
                encoding="utf-8",
            )
            fitted_summary.write_text(
                json.dumps(
                    _multiseed_summary(
                        run_dir=fitted_child,
                        bvi_attacker_cost=0.226,
                        policy_gap=0.3,
                        q_gap=1.5,
                        defend_probability=0.4,
                        defend_states=20,
                    )
                ),
                encoding="utf-8",
            )
            report_json = temp_dir / "diagnostic.json"
            report_md = temp_dir / "diagnostic.md"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "diagnose_routing_amq_cost_loss.py"),
                    "--normalized-summary",
                    str(normalized_summary),
                    "--fitted-summary",
                    str(fitted_summary),
                    "--json-output",
                    str(report_json),
                    "--markdown-output",
                    str(report_md),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["num_seeds"], 1)
            self.assertAlmostEqual(
                report["aggregate"]["mean_bvi_attacker_cost_delta"],
                0.002,
            )
            self.assertAlmostEqual(
                report["aggregate"]["mean_policy_gap_delta"],
                -0.5,
            )
            self.assertTrue(report["seed_diagnostics"][0]["largest_bucket_deltas"])
            markdown = report_md.read_text(encoding="utf-8")
            self.assertIn("Per-Seed AMQ Deltas", markdown)
            self.assertIn("Largest Bucket Changes", markdown)


def _multiseed_summary(
    *,
    run_dir: Path,
    bvi_attacker_cost: float,
    policy_gap: float,
    q_gap: float,
    defend_probability: float,
    defend_states: int,
) -> dict:
    return {
        "rows": [
            {
                "seed": 0,
                "algorithm_rows": [
                    {
                        "algorithm": "amq",
                        "run_dir": str(run_dir),
                        "average_cost_mean": 0.2,
                        "always_attack_average_cost_mean": 0.16,
                        "minimax_average_cost_mean": 0.21,
                        "bvi_attacker_average_cost_mean": bvi_attacker_cost,
                        "policy_gap_mean": policy_gap,
                        "q_reference_gap_mean": q_gap,
                        "defend_probability_mean": defend_probability,
                        "defend_states": defend_states,
                    },
                ],
            },
        ],
    }


def _child_summary(*, policy_gap: float, q_gap: float) -> dict:
    return {
        "policy_comparison": {
            "by_total_queue": [
                {
                    "total_queue": 0,
                    "p_defend_abs_gap_mean": policy_gap,
                    "p_defend_signed_gap_mean": policy_gap,
                },
            ],
            "by_imbalance": [
                {
                    "imbalance": 0,
                    "p_defend_abs_gap_mean": policy_gap,
                    "p_defend_signed_gap_mean": policy_gap,
                },
            ],
        },
        "q_diagnostic": {
            "by_total_queue": [
                {
                    "total_queue": 0,
                    "q_reference_abs_gap_mean": q_gap,
                    "amq_bellman_abs_residual_mean": 0.1,
                },
            ],
            "by_imbalance": [
                {
                    "imbalance": 0,
                    "q_reference_abs_gap_mean": q_gap,
                    "amq_bellman_abs_residual_mean": 0.1,
                },
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
