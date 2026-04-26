import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_queueing.features.service_rate_features import (
    service_rate_feature_dim,
    service_rate_features,
)


class ServiceRateFeatureTests(unittest.TestCase):
    def test_basic_quadratic_features(self):
        features = service_rate_features(
            state=3,
            attacker_action=1,
            defender_action=2,
            feature_set="basic_quadratic",
        )

        np.testing.assert_allclose(features, [1.0, 3.0, 9.0, 1.0, 2.0, 2.0])

    def test_action_interaction_dimension(self):
        self.assertEqual(service_rate_feature_dim("action_interaction"), 20)


if __name__ == "__main__":
    unittest.main()
