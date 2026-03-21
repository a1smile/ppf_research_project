# 导入 math、logging、dataclass 和类型标注工具。
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 导入 numpy 并命名为 np，用于数值计算。
import numpy as np


# 记录鲁棒投票过程统计信息的数据结构。
@dataclass
class RobustVoteStats:
    # 总投票次数。
    n_votes: int = 0
    # 观测到的最小权重。
    w_min: float = float("inf")
    # 观测到的最大权重。
    w_max: float = 0.0
    # 权重总和。
    w_sum: float = 0.0
    # top-m 截断后保留的候选数量累计值。
    trunc_used: int = 0
    # top-m 截断前候选总数量累计值。
    trunc_total: int = 0

    # 更新权重相关统计量。
    def update_weight(self, w: float) -> None:
        self.n_votes += 1
        self.w_min = min(self.w_min, w)
        self.w_max = max(self.w_max, w)
        self.w_sum += w

    # 更新 top-m 截断相关统计量。
    def update_trunc(self, used: int, total: int) -> None:
        self.trunc_used += used
        self.trunc_total += total

    # 汇总当前统计结果为字典。
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


# 鲁棒投票器，负责残差计算、核函数加权和 top-m 选择。
class RobustVoter:
    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None):
        # 保存配置和日志器。
        self.cfg = cfg
        self.logger = logger
        # 读取各项超参数。
        self.kernel = str(cfg.get("kernel", "gaussian")).lower()
        self.sigma = float(cfg.get("sigma", 0.6))
        self.tau = float(cfg.get("tau", 1.0))
        self.B = float(cfg.get("B", 1.0))
        self.top_m = int(cfg.get("top_m_per_bucket", 50))
        self.normalize_feature = bool(cfg.get("normalize_feature", True))

        # 检查核函数名称是否合法。
        assert self.kernel in ("trunc", "gaussian", "huber", "tukey")
        # 若有日志器，则记录初始化信息。
        if self.logger:
            self.logger.info(f"[RobustVote] enable=True kernel={self.kernel} sigma={self.sigma} tau={self.tau} B={self.B} top_m={self.top_m} normalize={self.normalize_feature}")

    # 根据残差计算对应的鲁棒权重。
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

        # 使用 B 对权重做上限裁剪。
        w = min(self.B, w)
        return float(w)

    # 计算场景特征和模型特征之间的残差。
    def residual(self, gs: np.ndarray, gm: np.ndarray, model_diameter: float) -> float:
        # 若不做归一化，则直接计算欧氏距离。
        if not self.normalize_feature:
            d = gs - gm
            return float(np.linalg.norm(d))
        # 否则按角度和模型直径做归一化缩放。
        scales = np.array([math.pi, math.pi, math.pi, max(1e-12, float(model_diameter))], dtype=np.float32)
        d = (gs - gm) / scales
        return float(np.linalg.norm(d))

    # 从残差数组中选出 top-m 个最小残差的索引。
    def select_top_m(self, residuals: np.ndarray) -> np.ndarray:
        """
        Returns indices of top-m smallest residuals.
        """
        # 实际保留数量不超过 top_m 和数组长度。
        m = min(self.top_m, residuals.shape[0])
        if m <= 0:
            return np.array([], dtype=np.int64)
        # 先用 argpartition 快速取得前 m 个候选。
        idx = np.argpartition(residuals, m - 1)[:m]
        # 再对这 m 个候选按残差从小到大排序。
        idx = idx[np.argsort(residuals[idx])]
        return idx

    # 将一个权重累加到投票累加器中，并更新统计信息。
    def vote(self, accumulator: np.ndarray, mr: int, bin_j: int, weight: float, stats: RobustVoteStats) -> None:
        accumulator[mr, bin_j] += weight
        stats.update_weight(weight)
