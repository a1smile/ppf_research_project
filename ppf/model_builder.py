# 导入 math、logging、dataclass、defaultdict 和类型标注工具。
import math
import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# 导入 numpy 和 open3d。
import numpy as np
import open3d as o3d

# 导入 PPF 特征计算、旋转计算和哈希结构。
from .ppf_features import compute_pair_features, angle_from_transformed_point, to_internal_feature_g, discretize_baseline
from .utils import rotation_matrix_from_axis_angle
from .rsmrq_hash import RSMRQHashTable, PPFEntry


# 基线哈希表，使用单一离散化键映射到候选条目列表。
class BaselineHashTable:
    """
    Baseline hash table: one dict keyed by baseline discretization.
    Provides query_buckets() to match RS-MRQ interface.
    """
    def __init__(self, angle_step: float, distance_step: float):
        # 保存角度和距离离散步长。
        self.angle_step = angle_step
        self.distance_step = distance_step
        # 建立从离散键到条目列表的映射表。
        self.table: Dict[Tuple[int, int, int, int], List[PPFEntry]] = defaultdict(list)

    # 将条目添加到离散后的哈希桶中。
    def add(self, g: np.ndarray, entry: PPFEntry) -> None:
        key = discretize_baseline(g, self.angle_step, self.distance_step)
        self.table[key].append(entry)

    # 根据特征查询对应哈希桶，返回与 RS-MRQ 接口兼容的桶列表。
    def query_buckets(self, g: np.ndarray) -> List[List[PPFEntry]]:
        key = discretize_baseline(g, self.angle_step, self.distance_step)
        return [self.table.get(key, [])]


# 定义 PPF 模型数据结构。
@dataclass
class PPFModel:
    # 角度离散步长。
    angle_step: float
    # 距离离散步长。
    distance_step: float
    # 模型参考点对的 alpha_m 查表。
    alpha_m: List[List[float]]
    # 模型直径。
    model_diameter: float
    # 每个参考点对应的预计算参考旋转矩阵。
    ref_R: List[np.ndarray]
    # 模型点坐标。
    pts: np.ndarray
    # 模型法向量。
    normals: np.ndarray
    # 哈希表对象，可能是基线哈希表，也可能是 RS-MRQ 哈希表。
    hash_table: Any  # BaselineHashTable or RSMRQHashTable
    # 是否启用 RS-MRQ。
    enable_rsmrq: bool
    # 合并方式。
    merge_mode: str


# 根据输入模型点云和配置构建 PPFModel。
def build_ppf_model(
    model_pcd: o3d.geometry.PointCloud,
    angle_step: float,
    distance_step: float,
    cfg: dict,
    logger: Optional[logging.Logger] = None
) -> PPFModel:
    # 读取模型点和法向量。
    pts = np.asarray(model_pcd.points).astype(np.float64)
    normals = np.asarray(model_pcd.normals).astype(np.float64)
    # 模型点数。
    N = len(pts)

    # 初始化 alpha_m 查表。
    alpha_m = [[0.0 for _ in range(N)] for __ in range(N)]
    # 初始化模型直径。
    model_diameter = 0.0

    # 为每个参考点预计算将法向量对齐到 +X 方向的旋转矩阵。
    ref_R: List[np.ndarray] = [None] * N  # type: ignore
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)

    for i in range(N):
        # 取第 i 个点的法向量并归一化。
        ni = normals[i]
        ni = ni / (np.linalg.norm(ni) + 1e-12)
        # 计算将该法向量旋转到 +X 所需的旋转轴。
        axis = np.cross(ni, ex)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            axis = ey.copy()
            axis_norm = 1.0
        axis /= axis_norm
        angle = math.acos(max(-1.0, min(1.0, float(ni @ ex))))
        ref_R[i] = rotation_matrix_from_axis_angle(axis, angle)

    # 读取增强模块开关。
    enable_rsmrq = bool(cfg.get("enable_rsmrq", False))
    enable_robust = bool(cfg.get("enable_robust_vote", False))

    # 默认合并方式为 union。
    merge_mode = "union"
    if enable_rsmrq:
        # 读取 RS-MRQ 配置并创建哈希表。
        rcfg = cfg.get("rsmrq", {})
        merge_mode = str(rcfg.get("merge_mode", "union"))
        hash_table = RSMRQHashTable(
            w_levels=rcfg.get("w_levels", []),
            T_tables=int(rcfg.get("T_tables", 4)),
            merge_mode=merge_mode,
            seed=int(rcfg.get("seed", cfg.get("seed", 0))),
            logger=logger
        )
    else:
        # 若未启用 RS-MRQ，则使用基线哈希表。
        hash_table = BaselineHashTable(angle_step, distance_step)

    # 枚举所有有序点对 (i, j)，其中 i != j。
    n_inserted = 0
    for i in range(N):
        pi = pts[i]
        ni = normals[i]
        Ri = ref_R[i]
        for j in range(N):
            if i == j:
                continue
            pj = pts[j]
            nj = normals[j]

            feat = compute_pair_features(pi, ni, pj, nj)
            if feat is None:
                continue
            f1, f2, f3, f4 = feat

            # 过滤距离过近的点对。
            if f4 < distance_step * 0.5:
                continue

            # 将原始特征转换为内部特征向量 g。
            g = to_internal_feature_g(f1, f2, f3, f4)
            # 若启用了鲁棒投票，则在条目中保留 g 用于残差计算。
            g_store = tuple(float(x) for x in g) if enable_robust else None

            # 构造哈希表条目并插入。
            entry = PPFEntry(mr=i, mi=j, g=g_store)
            hash_table.add(g, entry)
            n_inserted += 1

            # 计算并保存 alpha_m。
            pj_mg = Ri @ (pj - pi)
            angle = angle_from_transformed_point(pj_mg)
            alpha_m[i][j] = -angle

            # 更新模型直径。
            if f4 > model_diameter:
                model_diameter = f4

    # 若存在日志器，则输出模型构建摘要。
    if logger:
        logger.info(f"[Model] N={N}, inserted_pairs={n_inserted}, model_diameter={model_diameter:.6f}")
        logger.info(f"[Model] enable_rsmrq={enable_rsmrq}, merge_mode={merge_mode}, store_features_for_robust={enable_robust}")

    # 返回构建完成的 PPFModel。
    return PPFModel(
        angle_step=angle_step,
        distance_step=distance_step,
        alpha_m=alpha_m,
        model_diameter=model_diameter,
        ref_R=ref_R,
        pts=pts,
        normals=normals,
        hash_table=hash_table,
        enable_rsmrq=enable_rsmrq,
        merge_mode=merge_mode
    )
