import unittest
import numpy as np

from ppf.voting_robust import RobustVoter


class TestRobustVote(unittest.TestCase):
    def test_kernels(self):
        cfg = {
            "kernel": "trunc",
            "tau": 1.0,
            "sigma": 0.5,
            "B": 1.0,
            "top_m_per_bucket": 10,
            "normalize_feature": True
        }
        v = RobustVoter(cfg, logger=None)
        self.assertEqual(v.compute_weight(0.5), 1.0)
        self.assertEqual(v.compute_weight(1.5), 0.0)

        cfg["kernel"] = "gaussian"
        v = RobustVoter(cfg, logger=None)
        w0 = v.compute_weight(0.0)
        w1 = v.compute_weight(1.0)
        self.assertTrue(w0 >= w1)

        cfg["kernel"] = "huber"
        v = RobustVoter(cfg, logger=None)
        self.assertAlmostEqual(v.compute_weight(0.5), 1.0, places=6)
        self.assertTrue(v.compute_weight(2.0) < 1.0)

        cfg["kernel"] = "tukey"
        v = RobustVoter(cfg, logger=None)
        self.assertTrue(v.compute_weight(0.5) > 0.0)
        self.assertEqual(v.compute_weight(10.0), 0.0)

    def test_residual(self):
        cfg = {"kernel": "gaussian", "sigma": 1.0, "tau": 1.0, "B": 1.0, "top_m_per_bucket": 10, "normalize_feature": True}
        v = RobustVoter(cfg, logger=None)
        gs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        gm = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        r = v.residual(gs, gm, model_diameter=10.0)
        self.assertTrue(r > 0.0)


if __name__ == "__main__":
    unittest.main()
