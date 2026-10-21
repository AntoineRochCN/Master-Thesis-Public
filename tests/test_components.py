"""
Unit tests for Master-Thesis-Public core components.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from data.generate_dummy_data import generate_synthetic_market_data


try:
    import gymnasium
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class TestSyntheticData(unittest.TestCase):
    def test_market_data_shape(self):
        data = generate_synthetic_market_data(n_timesteps=500, seed=1)
        self.assertEqual(data.shape, (500, 20))
        self.assertFalse(np.any(np.isnan(data)))
        # Bids should be strictly less than asks
        bids = data[:, 0]
        asks = data[:, 8]
        self.assertTrue(np.all(bids < asks))


@unittest.skipUnless(HAS_GYMNASIUM, "gymnasium not installed in current environment")
class TestLatencyDistributions(unittest.TestCase):
    def test_latency_distribution_uniform(self):
        from utils_env import get_latency_prop, latency_merger
        dist = get_latency_prop("uniform", delta_max=5)
        self.assertEqual(len(dist.latency_range), 6)
        self.assertEqual(dist.max_latency, 4)
        self.assertAlmostEqual(float(np.sum(dist.latency_probabilities)), 1.0, places=5)

    def test_latency_distribution_gamma(self):
        from utils_env import get_latency_prop
        dist = get_latency_prop("gamma", delta_max=10, mean=2.0)
        self.assertEqual(len(dist.latency_range), 11)
        self.assertAlmostEqual(float(np.sum(dist.latency_probabilities)), 1.0, places=5)

    def test_latency_merger(self):
        from utils_env import get_latency_prop, latency_merger
        dist_a = get_latency_prop("uniform", delta_max=3)
        dist_b = get_latency_prop("uniform", delta_max=2)
        merged = latency_merger(dist_a, dist_b)
        self.assertEqual(len(merged.latency_probabilities), len(merged.latency_range) - 1)
        self.assertEqual(merged.latency_range[0], 0)
        self.assertEqual(merged.latency_range[-1], 5)


class TestDSACIndexing(unittest.TestCase):
    def test_qf_indexing_1d(self):
        # Verify that indexing qf_next_values with 1D index gives (B,) not (B, B)
        B = 32
        qf_next_values = np.random.randn(2, B, 2).astype(np.float32)
        normal_values = np.random.randn(2, B).astype(np.float32)

        idx_min = np.argmin(qf_next_values[..., 0], axis=0)  # (B,)
        self.assertEqual(idx_min.shape, (B,))

        target_Q_values = qf_next_values[idx_min, np.arange(B), 0]
        target_Z_values = normal_values[idx_min, np.arange(B)]

        self.assertEqual(target_Q_values.shape, (B,))
        self.assertEqual(target_Z_values.shape, (B,))


if __name__ == "__main__":
    unittest.main()
