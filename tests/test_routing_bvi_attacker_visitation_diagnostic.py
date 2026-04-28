import importlib.util
import unittest
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "diagnose_routing_bvi_attacker_visitation.py"
    spec = importlib.util.spec_from_file_location("bvi_attacker_visitation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RoutingBVIAttackerVisitationDiagnosticTests(unittest.TestCase):
    def test_summarize_weighted_policy_gap_uses_visit_fractions(self):
        module = _load_script_module()
        comparison_rows = [
            {
                "state": [0, 0, 0],
                "p_defend_amq": 0.2,
                "p_defend_bvi_reference": 0.0,
                "p_defend_signed_gap": 0.2,
                "p_defend_abs_gap": 0.2,
                "amq_over_defends": False,
                "amq_under_defends": False,
            },
            {
                "state": [1, 0, 0],
                "p_defend_amq": 0.0,
                "p_defend_bvi_reference": 0.8,
                "p_defend_signed_gap": -0.8,
                "p_defend_abs_gap": 0.8,
                "amq_over_defends": False,
                "amq_under_defends": True,
            },
            {
                "state": [0, 1, 0],
                "p_defend_amq": 1.0,
                "p_defend_bvi_reference": 0.0,
                "p_defend_signed_gap": 1.0,
                "p_defend_abs_gap": 1.0,
                "amq_over_defends": True,
                "amq_under_defends": False,
            },
        ]
        visitation_rows = [
            {"state": [0, 0, 0], "visit_count": 80, "visit_fraction": 0.8},
            {"state": [1, 0, 0], "visit_count": 20, "visit_fraction": 0.2},
        ]

        summary = module.summarize_weighted_policy_gap(
            comparison_rows,
            visitation_rows,
        )

        self.assertEqual(summary["num_states"], 3)
        self.assertEqual(summary["num_visited_states"], 2)
        self.assertEqual(summary["num_visited_over_defend_states"], 0)
        self.assertEqual(summary["num_visited_under_defend_states"], 1)
        self.assertAlmostEqual(summary["visit_weighted_abs_gap_mean"], 0.32)
        self.assertAlmostEqual(summary["visit_weighted_signed_gap_mean"], 0.0)
        self.assertAlmostEqual(summary["visited_p_defend_amq_mean"], 0.16)
        self.assertAlmostEqual(
            summary["visited_p_defend_bvi_reference_mean"],
            0.16,
        )
        self.assertEqual(summary["top_visited_gap_states"][0]["state"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
