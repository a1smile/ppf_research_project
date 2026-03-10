import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KDETrace:
    iters: int
    kde_values: list
    theta_init: float
    theta_refined: float


class KDEMeanShiftRefiner:
    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger
        self.use_embedding = bool(cfg.get("use_angle_embedding", True))
        self.top_k = int(cfg.get("top_k", 3))
        self.h = float(cfg.get("bandwidth_h", 0.35))
        self.max_iter = int(cfg.get("max_iter", 50))
        self.tol = float(cfg.get("tol", 1e-4))

        if self.logger:
            self.logger.info(f"[KDERefine] enable=True embedding={self.use_embedding} top_k={self.top_k} h={self.h} max_iter={self.max_iter} tol={self.tol}")

    def _kde_2d(self, x: np.ndarray, xs: np.ndarray, w: np.ndarray) -> float:
        # x: (2,), xs:(N,2)
        d2 = np.sum((xs - x[None, :]) ** 2, axis=1)
        k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
        return float(np.sum(w * k))

    def _meanshift_2d(self, x: np.ndarray, xs: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
        d2 = np.sum((xs - x[None, :]) ** 2, axis=1)
        k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
        wk = w * k
        denom = float(np.sum(wk)) + 1e-12
        x_next = (wk[:, None] * xs).sum(axis=0) / denom
        kde_val = float(np.sum(wk))
        return x_next, kde_val

    def refine(self, thetas: np.ndarray, weights: np.ndarray, theta_init: float) -> Tuple[float, KDETrace]:
        """
        thetas: (N,) in [-pi, pi]
        weights: (N,) non-negative
        theta_init: initial angle (bin center)
        Returns refined theta and trace.
        """
        if thetas.size == 0:
            return theta_init, KDETrace(iters=0, kde_values=[], theta_init=theta_init, theta_refined=theta_init)

        w = weights.astype(np.float32)
        w = np.maximum(w, 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return theta_init, KDETrace(iters=0, kde_values=[], theta_init=theta_init, theta_refined=theta_init)
        w = w / w_sum

        kde_values = []

        if self.use_embedding:
            xs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1).astype(np.float32)  # (N,2)
            x = np.array([math.cos(theta_init), math.sin(theta_init)], dtype=np.float32)

            prev_kde = self._kde_2d(x, xs, w)
            kde_values.append(prev_kde)

            it = 0
            for it in range(1, self.max_iter + 1):
                x_next, _ = self._meanshift_2d(x, xs, w)
                kde_val = self._kde_2d(x_next, xs, w)
                kde_values.append(kde_val)
                if self.logger:
                    self.logger.info(f"[KDERefine] iter={it} kde={kde_val:.6f}")
                if float(np.linalg.norm(x_next - x)) < self.tol:
                    x = x_next
                    break
                x = x_next

            theta_ref = math.atan2(float(x[1]), float(x[0]))
            trace = KDETrace(iters=it, kde_values=kde_values, theta_init=theta_init, theta_refined=theta_ref)
            return theta_ref, trace

        # 1D fallback: unwrap near init
        # map theta to closest representation near theta_init
        def unwrap(theta):
            a = theta
            while a - theta_init > math.pi:
                a -= 2.0 * math.pi
            while a - theta_init < -math.pi:
                a += 2.0 * math.pi
            return a

        ts = np.array([unwrap(t) for t in thetas], dtype=np.float32)
        x = float(theta_init)
        it = 0
        for it in range(1, self.max_iter + 1):
            d2 = (ts - x) ** 2
            k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
            wk = w * k
            denom = float(np.sum(wk)) + 1e-12
            x_next = float(np.sum(wk * ts) / denom)
            # kde value in 1D (up to constant)
            kde_val = float(np.sum(wk))
            kde_values.append(kde_val)
            if self.logger:
                self.logger.info(f"[KDERefine] iter={it} kde={kde_val:.6f}")
            if abs(x_next - x) < self.tol:
                x = x_next
                break
            x = x_next

        theta_ref = math.atan2(math.sin(x), math.cos(x))
        trace = KDETrace(iters=it, kde_values=kde_values, theta_init=theta_init, theta_refined=theta_ref)
        return theta_ref, trace
