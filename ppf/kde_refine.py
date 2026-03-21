# 导入 math 模块，用于三角函数和角度运算。
import math
# 导入 logging 模块，用于日志输出。
import logging
# 导入 dataclass 装饰器，用于定义轨迹数据结构。
from dataclasses import dataclass
# 导入类型标注工具。
from typing import Optional, Tuple

# 导入 numpy 并命名为 np，用于数值计算。
import numpy as np


# 定义 KDE 精化过程的轨迹记录结构。
@dataclass
class KDETrace:
    # 实际迭代次数。
    iters: int
    # 每次迭代对应的 KDE 值序列。
    kde_values: list
    # 初始角度。
    theta_init: float
    # 精化后的角度。
    theta_refined: float


# 基于 KDE 和均值漂移的角度精化器。
class KDEMeanShiftRefiner:
    def __init__(self, cfg: dict, logger: Optional[logging.Logger] = None):
        # 保存原始配置。
        self.cfg = cfg
        # 保存日志器。
        self.logger = logger
        # 是否使用二维角度嵌入表示。
        self.use_embedding = bool(cfg.get("use_angle_embedding", True))
        # 后续使用的 top-k 候选数量。
        self.top_k = int(cfg.get("top_k", 3))
        # KDE 带宽参数。
        self.h = float(cfg.get("bandwidth_h", 0.35))
        # 最大迭代次数。
        self.max_iter = int(cfg.get("max_iter", 50))
        # 收敛阈值。
        self.tol = float(cfg.get("tol", 1e-4))

        # 如果提供了日志器，则输出当前精化器配置。
        if self.logger:
            self.logger.info(f"[KDERefine] enable=True embedding={self.use_embedding} top_k={self.top_k} h={self.h} max_iter={self.max_iter} tol={self.tol}")

    # 在二维嵌入空间中计算给定点的 KDE 值。
    def _kde_2d(self, x: np.ndarray, xs: np.ndarray, w: np.ndarray) -> float:
        # 计算样本点与当前点之间的平方距离。
        d2 = np.sum((xs - x[None, :]) ** 2, axis=1)
        # 按高斯核公式计算每个样本的核值。
        k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
        # 对加权核值求和并返回。
        return float(np.sum(w * k))

    # 在二维嵌入空间中执行一次均值漂移更新。
    def _meanshift_2d(self, x: np.ndarray, xs: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
        # 计算当前点到各样本点的平方距离。
        d2 = np.sum((xs - x[None, :]) ** 2, axis=1)
        # 计算高斯核值。
        k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
        # 将样本权重与核值相乘。
        wk = w * k
        # 计算归一化分母。
        denom = float(np.sum(wk)) + 1e-12
        # 计算更新后的均值漂移位置。
        x_next = (wk[:, None] * xs).sum(axis=0) / denom
        # 计算当前位置对应的 KDE 值。
        kde_val = float(np.sum(wk))
        # 返回更新后的位置和 KDE 值。
        return x_next, kde_val

    # 对给定角度样本和权重执行 KDE 精化。
    def refine(self, thetas: np.ndarray, weights: np.ndarray, theta_init: float) -> Tuple[float, KDETrace]:
        """
        thetas: (N,) in [-pi, pi]
        weights: (N,) non-negative
        theta_init: initial angle (bin center)
        Returns refined theta and trace.
        """
        # 如果没有任何角度样本，则直接返回初始角度。
        if thetas.size == 0:
            return theta_init, KDETrace(iters=0, kde_values=[], theta_init=theta_init, theta_refined=theta_init)

        # 将权重转换为 float32。
        w = weights.astype(np.float32)
        # 将负权重截断为 0。
        w = np.maximum(w, 0.0)
        # 计算权重总和。
        w_sum = float(np.sum(w))
        # 若总权重过小，则直接返回初始角度。
        if w_sum <= 1e-12:
            return theta_init, KDETrace(iters=0, kde_values=[], theta_init=theta_init, theta_refined=theta_init)
        # 将权重归一化。
        w = w / w_sum

        # 用于保存每一步的 KDE 值。
        kde_values = []

        # 若启用角度嵌入，则在二维单位圆空间中执行精化。
        if self.use_embedding:
            # 将角度映射到二维单位圆表示。
            xs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1).astype(np.float32)  # (N,2)
            # 构造初始点的二维表示。
            x = np.array([math.cos(theta_init), math.sin(theta_init)], dtype=np.float32)

            # 计算初始点的 KDE 值。
            prev_kde = self._kde_2d(x, xs, w)
            kde_values.append(prev_kde)

            # 初始化迭代计数。
            it = 0
            # 进行均值漂移迭代。
            for it in range(1, self.max_iter + 1):
                # 执行一步二维均值漂移更新。
                x_next, _ = self._meanshift_2d(x, xs, w)
                # 计算更新后位置的 KDE 值。
                kde_val = self._kde_2d(x_next, xs, w)
                kde_values.append(kde_val)
                # 记录日志。
                if self.logger:
                    self.logger.info(f"[KDERefine] iter={it} kde={kde_val:.6f}")
                # 若更新幅度小于阈值，则认为收敛。
                if float(np.linalg.norm(x_next - x)) < self.tol:
                    x = x_next
                    break
                x = x_next

            # 将二维嵌入点重新映射回角度。
            theta_ref = math.atan2(float(x[1]), float(x[0]))
            # 构造轨迹对象。
            trace = KDETrace(iters=it, kde_values=kde_values, theta_init=theta_init, theta_refined=theta_ref)
            # 返回精化后的角度和轨迹。
            return theta_ref, trace

        # 若不使用嵌入，则退化为一维角度空间的均值漂移。
        # 将角度展开到接近 theta_init 的表示形式。
        def unwrap(theta):
            a = theta
            while a - theta_init > math.pi:
                a -= 2.0 * math.pi
            while a - theta_init < -math.pi:
                a += 2.0 * math.pi
            return a

        # 对所有角度执行展开。
        ts = np.array([unwrap(t) for t in thetas], dtype=np.float32)
        # 将当前估计初始化为 theta_init。
        x = float(theta_init)
        # 初始化迭代计数。
        it = 0
        # 进行一维均值漂移迭代。
        for it in range(1, self.max_iter + 1):
            # 计算与当前估计的平方距离。
            d2 = (ts - x) ** 2
            # 计算高斯核值。
            k = np.exp(-d2 / (2.0 * self.h * self.h + 1e-12))
            # 合成加权核值。
            wk = w * k
            # 计算归一化分母。
            denom = float(np.sum(wk)) + 1e-12
            # 计算下一步均值漂移位置。
            x_next = float(np.sum(wk * ts) / denom)
            # 计算一维情形下的 KDE 值。
            kde_val = float(np.sum(wk))
            kde_values.append(kde_val)
            # 记录日志。
            if self.logger:
                self.logger.info(f"[KDERefine] iter={it} kde={kde_val:.6f}")
            # 若更新幅度小于阈值，则认为收敛。
            if abs(x_next - x) < self.tol:
                x = x_next
                break
            x = x_next

        # 将结果重新映射回 [-pi, pi]。
        theta_ref = math.atan2(math.sin(x), math.cos(x))
        # 构造并返回轨迹对象。
        trace = KDETrace(iters=it, kde_values=kde_values, theta_init=theta_init, theta_refined=theta_ref)
        return theta_ref, trace
