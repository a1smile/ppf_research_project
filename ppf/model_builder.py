import math
import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import open3d as o3d

from .ppf_features import compute_pair_features, angle_from_transformed_point, to_internal_feature_g, discretize_baseline
from .utils import rotation_matrix_from_axis_angle
from .rsmrq_hash import RSMRQHashTable, PPFEntry


class BaselineHashTable:
    """
    Baseline hash table: one dict keyed by baseline discretization.
    Provides query_buckets() to match RS-MRQ interface.
    """
    def __init__(self, angle_step: float, distance_step: float):
        self.angle_step = angle_step
        self.distance_step = distance_step
        self.table: Dict[Tuple[int, int, int, int], List[PPFEntry]] = defaultdict(list)

    def add(self, g: np.ndarray, entry: PPFEntry) -> None:
        key = discretize_baseline(g, self.angle_step, self.distance_step)
        self.table[key].append(entry)

    def query_buckets(self, g: np.ndarray) -> List[List[PPFEntry]]:
        key = discretize_baseline(g, self.angle_step, self.distance_step)
        return [self.table.get(key, [])]


@dataclass
class PPFModel:
    angle_step: float
    distance_step: float
    alpha_m: List[List[float]]
    model_diameter: float
    ref_R: List[np.ndarray]
    pts: np.ndarray
    normals: np.ndarray
    hash_table: Any  # BaselineHashTable or RSMRQHashTable
    enable_rsmrq: bool
    merge_mode: str


def build_ppf_model(
    model_pcd: o3d.geometry.PointCloud,
    angle_step: float,
    distance_step: float,
    cfg: dict,
    logger: Optional[logging.Logger] = None
) -> PPFModel:
    pts = np.asarray(model_pcd.points).astype(np.float64)
    normals = np.asarray(model_pcd.normals).astype(np.float64)
    N = len(pts)

    alpha_m = [[0.0 for _ in range(N)] for __ in range(N)]
    model_diameter = 0.0

    # Precompute R_mg for each reference (align normal to +X)
    ref_R: List[np.ndarray] = [None] * N  # type: ignore
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)

    for i in range(N):
        ni = normals[i]
        ni = ni / (np.linalg.norm(ni) + 1e-12)
        axis = np.cross(ni, ex)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            axis = ey.copy()
            axis_norm = 1.0
        axis /= axis_norm
        angle = math.acos(max(-1.0, min(1.0, float(ni @ ex))))
        ref_R[i] = rotation_matrix_from_axis_angle(axis, angle)

    enable_rsmrq = bool(cfg.get("enable_rsmrq", False))
    enable_robust = bool(cfg.get("enable_robust_vote", False))

    merge_mode = "union"
    if enable_rsmrq:
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
        hash_table = BaselineHashTable(angle_step, distance_step)

    # Enumerate all ordered pairs (i,j), i!=j
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

            if f4 < distance_step * 0.5:
                continue

            g = to_internal_feature_g(f1, f2, f3, f4)
            g_store = tuple(float(x) for x in g) if enable_robust else None

            entry = PPFEntry(mr=i, mi=j, g=g_store)
            hash_table.add(g, entry)
            n_inserted += 1

            # alpha_m computation
            pj_mg = Ri @ (pj - pi)
            angle = angle_from_transformed_point(pj_mg)
            alpha_m[i][j] = -angle

            if f4 > model_diameter:
                model_diameter = f4

    if logger:
        logger.info(f"[Model] N={N}, inserted_pairs={n_inserted}, model_diameter={model_diameter:.6f}")
        logger.info(f"[Model] enable_rsmrq={enable_rsmrq}, merge_mode={merge_mode}, store_features_for_robust={enable_robust}")

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
