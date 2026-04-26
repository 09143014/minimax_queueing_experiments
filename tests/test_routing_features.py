import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.features.routing_features import routing_feature_dim, routing_features


class RoutingFeatureTests(unittest.TestCase):
    def test_basic_feature_dimension_includes_queue_lengths(self):
        features = routing_features((1, 2, 3), attacker_action=1, defender_action=0)

        self.assertEqual(features.shape[0], routing_feature_dim(3))
        self.assertEqual(routing_feature_dim(3), 10)


if __name__ == "__main__":
    unittest.main()
