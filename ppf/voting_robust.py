import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class RobustVoteStats:
    n_votes: int = 0
    w_min: float = float("inf")
    w_max: float = 0.0
    w_sum: float = 0.0
    trunc_used: int = 0
    trunc_total: int = 0

    def update_weight(self, w: float) -> None:
        self.n_votes += 1
        self.w_min = min(self.w_min, w)
        self.w_max = max(self.w_max, w)
        self.w_sum += w

    def update_trunc(self, used: int, total: int) -> None:
        self.trunc_used += used
        self.trunc_total += total

    def summary(self) -> dict:
        mean = self.w_sum / max(1, self.n_votes)
        trunc_ratio = self.trunc_used / max(1, self.trunc_total)
        return {
            "n_votes": self.n_votes,
            "w_min": self.w_min if self.n_votes > 0 else 0.0,
            "w_mean": mean,
            "w_max": self.w_max,
            "topm_used": self.trunc_used,
            "topm_total": self.trunc_total,
            "topm_ratio": trunc_ratio
        }


class RobustVoter:
    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger
        self.kernel = str(cfg.get("kernel", "gaussian")).lower()
        self.sigma = float(cfg.get("sigma", 0.6))
        self.tau = float(cfg.get("tau", 1.0))
        self.B = float(cfg.get("B", 1.0))
        self.top_m = int(cfg.get("top_m_per_bucket", 50))
        self.normalize_feature = bool(cfg.get("normalize_feature", True))

        assert self.kernel in ("trunc", "gaussian", "huber", "tukey")
        if self.logger:
            self.logger.info(f"[RobustVote] enable=True kernel={self.kernel} sigma={self.sigma} tau={self.tau} B={self.B} top_m={self.top_m} normalize={self.normalize_feature}")

    def compute_weight(self, r: float) -> float:
        if self.kernel == "trunc":
            w = 1.0 if r <= self.tau else 0.0
        elif self.kernel == "gaussian":
            w = math.exp(-(r * r) / (2.0 * self.sigma * self.sigma + 1e-12))
        elif self.kernel == "huber":
            w = 1.0 if r <= self.tau else (self.tau / (r + 1e-12))
        else:  # tukey
            if r <= self.tau:
                t = 1.0 - (r / (self.tau + 1e-12)) ** 2
                w = t * t
            else:
                w = 0.0

        w = min(self.B, w)
        return float(w)

    def residual(self, gs: np.ndarray, gm: np.ndarray, model_diameter: float) -> float:
        if not self.normalize_feature:
            d = gs - gm
            return float(np.linalg.norm(d))
        scales = np.array([math.pi, math.pi, math.pi, max(1e-12, float(model_diameter))], dtype=np.float32)
        d = (gs - gm) / scales
        return float(np.linalg.norm(d))

    def select_top_m(self, residuals: np.ndarray) -> np.ndarray:
        """
        Returns indices of top-m smallest residuals.
        """
        m = min(self.top_m, residuals.shape[0])
        if m <= 0:
            return np.array([], dtype=np.int64)
        # argpartition for speed
        idx = np.argpartition(residuals, m - 1)[:m]
        # sort these m by residual
        idx = idx[np.argsort(residuals[idx])]
        return idx

    def vote(self, accumulator: np.ndarray, mr: int, bin_j: int, weight: float, stats: RobustVoteStats) -> None:
        accumulator[mr, bin_j] += weight
        stats.update_weight(weight)
