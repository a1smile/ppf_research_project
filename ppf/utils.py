import os
import sys
import math
import json
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def wrap_to_pi(a: float) -> float:
    # wrap to [-pi, pi]
    while a < -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = ax
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                  [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                  [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=float)
    return R


def make_affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_affine(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_affine(R_inv, t_inv)


def compose_affine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


def so3_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    tr = float(np.trace(R))
    val = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    return math.acos(val)


@dataclass
class Timer:
    name: str
    logger: Optional[logging.Logger] = None
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        if self.logger:
            self.logger.info(f"[TIMER] {self.name}: {self.elapsed:.6f} sec")


def setup_logger(log_path: str, level: int = logging.INFO) -> logging.Logger:
    ensure_dir(os.path.dirname(log_path))
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger
