# 导入 math 模块和类型标注工具。
import math
from typing import Optional, Tuple

# 导入 numpy 并命名为 np，用于向量计算。
import numpy as np


# 计算一对点及其法向量的 PPF 风格特征。
def compute_pair_features(
    p1: np.ndarray, n1: np.ndarray,
    p2: np.ndarray, n2: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Baseline-compatible PFH-style pair features:
    Returns (f1, f2, f3, f4) or None if degenerate.
    f1 in [-pi, pi]; f2, f3 in [-1, 1]; f4 >= 0.
    """
    # 计算点对差向量。
    dp = p2 - p1
    # 计算点对距离。
    f4 = float(np.linalg.norm(dp))
    # 若距离过小，则视为退化情况。
    if f4 <= 1e-9:
        return None

    # 归一化方向向量。
    d = dp / f4
    # 归一化两个法向量。
    n1n = n1 / (np.linalg.norm(n1) + 1e-12)
    n2n = n2 / (np.linalg.norm(n2) + 1e-12)

    # 若参考法向量与方向向量几乎平行，则视为退化情况。
    if abs(float(n1n @ d)) > 0.999:
        return None

    # 构造局部坐标系 u, v, w。
    u = n1n
    v = np.cross(u, d)
    nv = np.linalg.norm(v)
    # 若 v 退化，则无法构造稳定坐标系。
    if nv <= 1e-12:
        return None
    v /= nv
    w = np.cross(u, v)

    # 计算三个角度/方向特征分量。
    f1 = math.atan2(float(w @ n2n), float(u @ n2n))
    f2 = float(v @ n2n)
    f3 = float(u @ d)
    # 返回 PPF 特征四元组。
    return (f1, f2, f3, f4)


# 根据变换后的点位置计算参考角度。
def angle_from_transformed_point(vec_yzx: np.ndarray) -> float:
    # 取向量在 y 和 z 方向的分量。
    y = float(vec_yzx[1])
    z = float(vec_yzx[2])
    # 根据 y-z 平面位置计算角度。
    ang = math.atan2(-z, y)
    # 按基线实现的规则进行符号修正。
    if math.sin(ang) * z < 0.0:
        ang *= -1.0
    # 返回角度。
    return ang


# 将原始特征转换为内部使用的特征向量 g。
def to_internal_feature_g(f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """
    Internal feature vector g used for hashing/residuals:
    g1 = f1 + pi          in [0, 2pi)
    g2 = acos(clamp(f2))  in [0, pi]
    g3 = acos(clamp(f3))  in [0, pi]
    g4 = f4               distance
    """
    # 对 f2、f3 做夹取，避免 acos 数值越界。
    f2c = max(-1.0, min(1.0, f2))
    f3c = max(-1.0, min(1.0, f3))
    # 构造四维内部特征。
    g1 = f1 + math.pi
    g2 = math.acos(f2c)
    g3 = math.acos(f3c)
    g4 = f4
    # 返回 float32 类型的内部特征向量。
    return np.array([g1, g2, g3, g4], dtype=np.float32)


# 按基线规则对内部特征 g 进行离散化。
def discretize_baseline(g: np.ndarray, angle_step: float, distance_step: float) -> tuple[int, int, int, int]:
    """
    Baseline-compatible discretization:
    k1 = floor(g1/angle_step), k2=floor(g2/angle_step), k3=floor(g3/angle_step), k4=floor(g4/distance_step)
    """
    # 分别对四个维度做 floor 离散化。
    k1 = int(math.floor(float(g[0]) / angle_step))
    k2 = int(math.floor(float(g[1]) / angle_step))
    k3 = int(math.floor(float(g[2]) / angle_step))
    k4 = int(math.floor(float(g[3]) / distance_step))
    # 返回离散后的四维索引键。
    return (k1, k2, k3, k4)
