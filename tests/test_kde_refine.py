import unittest
import numpy as np

from ppf.kde_refine import KDEMeanShiftRefiner


class TestKDERefine(unittest.TestCase):
    def test_refine_runs(self):
        cfg = {
            "use_angle_embedding": True,
            "top_k": 3,
            "bandwidth_h": 0.3,
            "max_iter": 50,
            "tol": 1e-6
        }
        r = KDEMeanShiftRefiner(cfg, logger=None)
        thetas = np.array([0.0, 0.05, -0.03, 3.12, -3.10], dtype=np.float32)  # wrap-around samples
        weights = np.array([1, 1, 1, 0.2, 0.2], dtype=np.float32)
        theta_ref, trace = r.refine(thetas, weights, theta_init=0.0)
        self.assertTrue(abs(theta_ref) < 0.2)
        self.assertTrue(trace.iters >= 1)

    def test_monotonic_trace(self):
        cfg = {"use_angle_embedding": True, "bandwidth_h": 0.4, "max_iter": 20, "tol": 1e-8}
        r = KDEMeanShiftRefiner(cfg, logger=None)
        thetas = np.array([0.0, 0.1, 0.12, 0.08], dtype=np.float32)
        weights = np.ones_like(thetas)
        _, trace = r.refine(thetas, weights, theta_init=0.05)
        # empirical monotonic check (non-decreasing)
        vals = trace.kde_values
        for i in range(1, len(vals)):
            self.assertTrue(vals[i] >= vals[i-1] - 1e-6)


if __name__ == "__main__":
    unittest.main()
