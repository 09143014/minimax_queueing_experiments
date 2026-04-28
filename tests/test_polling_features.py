import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.features.polling_features import (
    polling_feature_dim,
    polling_features,
)


class PollingFeatureTests(unittest.TestCase):
    def test_basic_feature_dimension(self):
        features = polling_features((1, 2, 0), 1, 0, "basic")

        self.assertEqual(features.shape[0], polling_feature_dim(2, "basic"))

    def test_action_interaction_feature_dimension(self):
        features = polling_features((1, 2, 0), 1, 1, "action_interaction")

        self.assertEqual(features.shape[0], polling_feature_dim(2, "action_interaction"))


if __name__ == "__main__":
    unittest.main()
