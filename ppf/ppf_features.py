import math
from typing import Optional, Tuple

import numpy as np


def compute_pair_features(
    p1: np.ndarray, n1: np.ndarray,
    p2: np.ndarray, n2: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Baseline-compatible PFH-style pair features:
    Returns (f1, f2, f3, f4) or None if degenerate.
    f1 in [-pi, pi]; f2, f3 in [-1, 1]; f4 >= 0.
    """
    dp = p2 - p1
    f4 = float(np.linalg.norm(dp))
    if f4 <= 1e-9:
        return None

    d = dp / f4
    n1n = n1 / (np.linalg.norm(n1) + 1e-12)
    n2n = n2 / (np.linalg.norm(n2) + 1e-12)

    if abs(float(n1n @ d)) > 0.999:
        return None

    u = n1n
    v = np.cross(u, d)
    nv = np.linalg.norm(v)
    if nv <= 1e-12:
        return None
    v /= nv
    w = np.cross(u, v)

    f1 = math.atan2(float(w @ n2n), float(u @ n2n))
    f2 = float(v @ n2n)
    f3 = float(u @ d)
    return (f1, f2, f3, f4)


def angle_from_transformed_point(vec_yzx: np.ndarray) -> float:
    y = float(vec_yzx[1])
    z = float(vec_yzx[2])
    ang = math.atan2(-z, y)
    if math.sin(ang) * z < 0.0:
        ang *= -1.0
    return ang


def to_internal_feature_g(f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """
    Internal feature vector g used for hashing/residuals:
    g1 = f1 + pi          in [0, 2pi)
    g2 = acos(clamp(f2))  in [0, pi]
    g3 = acos(clamp(f3))  in [0, pi]
    g4 = f4               distance
    """
    f2c = max(-1.0, min(1.0, f2))
    f3c = max(-1.0, min(1.0, f3))
    g1 = f1 + math.pi
    g2 = math.acos(f2c)
    g3 = math.acos(f3c)
    g4 = f4
    return np.array([g1, g2, g3, g4], dtype=np.float32)


def discretize_baseline(g: np.ndarray, angle_step: float, distance_step: float) -> tuple[int, int, int, int]:
    """
    Baseline-compatible discretization:
    k1 = floor(g1/angle_step), k2=floor(g2/angle_step), k3=floor(g3/angle_step), k4=floor(g4/distance_step)
    """
    k1 = int(math.floor(float(g[0]) / angle_step))
    k2 = int(math.floor(float(g[1]) / angle_step))
    k3 = int(math.floor(float(g[2]) / angle_step))
    k4 = int(math.floor(float(g[3]) / distance_step))
    return (k1, k2, k3, k4)
