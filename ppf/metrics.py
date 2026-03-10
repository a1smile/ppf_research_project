import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import open3d as o3d

from .utils import so3_distance, make_affine, invert_affine, compose_affine, wrap_to_pi


def rotation_translation_error(T_pred: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    rot_err = so3_distance(R_gt, R_pred) * 180.0 / math.pi
    trans_err = float(np.linalg.norm(t_gt - t_pred))
    return rot_err, trans_err


def add_metric(model_pts: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    Pg = (T_gt[:3, :3] @ model_pts.T).T + T_gt[:3, 3]
    return float(np.mean(np.linalg.norm(Pp - Pg, axis=1)))


def add_s_metric(model_pts: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    Pg = (T_gt[:3, :3] @ model_pts.T).T + T_gt[:3, 3]
    # nearest neighbor from Pp to Pg
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pg.astype(np.float64))
    kd = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(Pp.shape[0]):
        _, idx, dist2 = kd.search_knn_vector_3d(Pp[i], 1)
        dists.append(math.sqrt(dist2[0]) if len(dist2) > 0 else 1e9)
    return float(np.mean(np.array(dists, dtype=np.float64)))


def inlier_ratio_model_to_scene(model_pts_transformed: np.ndarray, scene_pts: np.ndarray, radius: float) -> float:
    """
    Fraction of transformed model points that have a neighbor in scene within radius.
    """
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts.astype(np.float64))
    kd = o3d.geometry.KDTreeFlann(scene_pcd)
    hit = 0
    for i in range(model_pts_transformed.shape[0]):
        k, _, _ = kd.search_radius_vector_3d(model_pts_transformed[i], radius)
        if k > 0:
            hit += 1
    return float(hit) / max(1, model_pts_transformed.shape[0])


def compute_metrics(
    model_pts: np.ndarray,
    scene_pts: np.ndarray,
    T_pred: np.ndarray,
    T_gt: Optional[np.ndarray] = None,
    inlier_radius: float = 5.0
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if T_gt is None:
        out["ADD"] = float("nan")
        out["ADD_S"] = float("nan")
        out["rotation_error_deg"] = float("nan")
        out["translation_error"] = float("nan")
    else:
        out["ADD"] = add_metric(model_pts, T_pred, T_gt)
        out["ADD_S"] = add_s_metric(model_pts, T_pred, T_gt)
        rerr, terr = rotation_translation_error(T_pred, T_gt)
        out["rotation_error_deg"] = rerr
        out["translation_error"] = terr

    # always compute inlier ratio
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    out["inlier_ratio"] = inlier_ratio_model_to_scene(Pp, scene_pts, radius=inlier_radius)
    return out
