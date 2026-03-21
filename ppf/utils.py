# 导入常用标准库模块和类型工具。
import os
import sys
import math
import json
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

# 导入 numpy 和 yaml。
import numpy as np
import yaml


# 设置 Python 和 numpy 的随机种子。
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# 读取 YAML 文件并返回解析后的对象。
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 确保目录存在，若不存在则自动创建。
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# 将字典对象保存为 JSON 文件。
def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# 将角度包装到 [-pi, pi] 区间内。
def wrap_to_pi(a: float) -> float:
    while a < -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


# 根据旋转轴和旋转角构造旋转矩阵。
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


# 根据旋转矩阵和平移向量构造 4x4 仿射变换矩阵。
def make_affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# 求 4x4 仿射变换矩阵的逆。
def invert_affine(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_affine(R_inv, t_inv)


# 计算两个仿射变换矩阵的复合。
def compose_affine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


# 将一组点应用给定仿射变换。
def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


# 计算两个旋转矩阵之间的 SO(3) 角距离。
def so3_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    tr = float(np.trace(R))
    val = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    return math.acos(val)


# 定义一个可用 with 语法记录耗时的计时器。
@dataclass
class Timer:
    # 计时器名称。
    name: str
    # 可选日志器。
    logger: Optional[logging.Logger] = None
    # 起始时间。
    start: float = 0.0
    # 已耗时间。
    elapsed: float = 0.0

    # 进入上下文时记录起始时间。
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    # 退出上下文时计算耗时并可选写入日志。
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        if self.logger:
            self.logger.info(f"[TIMER] {self.name}: {self.elapsed:.6f} sec")


# 创建同时输出到文件和标准输出的日志器。
def setup_logger(log_path: str, level: int = logging.INFO) -> logging.Logger:
    ensure_dir(os.path.dirname(log_path))
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    # 清空已有 handler，避免重复输出。
    logger.handlers.clear()

    # 定义统一日志格式。
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # 创建文件日志 handler。
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 创建控制台日志 handler。
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 关闭向父级 logger 传播。
    logger.propagate = False
    # 返回配置完成的 logger。
    return logger
