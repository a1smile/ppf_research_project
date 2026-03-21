# 导入 math 模块，用于数学计算。
import math
# 导入类型标注工具。
from typing import Optional, Dict, Any, Tuple

# 导入 numpy 和 open3d，用于数值计算和点云近邻搜索。
import numpy as np
import open3d as o3d

# 从工具模块导入旋转距离函数。
from .utils import so3_distance, make_affine, invert_affine, compose_affine, wrap_to_pi


# 计算预测位姿与 GT 位姿之间的旋转误差和位移误差。
def rotation_translation_error(T_pred: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    # 提取预测旋转和平移。
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]
    # 提取 GT 旋转和平移。
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    # 计算旋转误差并转换为角度制。
    rot_err = so3_distance(R_gt, R_pred) * 180.0 / math.pi
    # 计算平移误差。
    trans_err = float(np.linalg.norm(t_gt - t_pred))
    # 返回旋转误差和平移误差。
    return rot_err, trans_err


# 计算 ADD 指标。
def add_metric(model_pts: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    # 用预测位姿变换模型点。
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    # 用 GT 位姿变换模型点。
    Pg = (T_gt[:3, :3] @ model_pts.T).T + T_gt[:3, 3]
    # 返回两组点对应距离的平均值。
    return float(np.mean(np.linalg.norm(Pp - Pg, axis=1)))


# 计算 ADD-S 指标。
def add_s_metric(model_pts: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    # 用预测位姿变换模型点。
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    # 用 GT 位姿变换模型点。
    Pg = (T_gt[:3, :3] @ model_pts.T).T + T_gt[:3, 3]
    # 通过最近邻方式计算对称匹配距离。
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pg.astype(np.float64))
    kd = o3d.geometry.KDTreeFlann(pcd)
    # 保存每个预测点到 GT 点集的最近邻距离。
    dists = []
    for i in range(Pp.shape[0]):
        # 搜索单个最近邻。
        _, idx, dist2 = kd.search_knn_vector_3d(Pp[i], 1)
        # 若找到邻居则取距离，否则给一个极大值。
        dists.append(math.sqrt(dist2[0]) if len(dist2) > 0 else 1e9)
    # 返回平均最近邻距离。
    return float(np.mean(np.array(dists, dtype=np.float64)))


# 计算变换后模型点到场景点的内点比例。
def inlier_ratio_model_to_scene(model_pts_transformed: np.ndarray, scene_pts: np.ndarray, radius: float) -> float:
    """
    Fraction of transformed model points that have a neighbor in scene within radius.
    """
    # 将场景点构造成 Open3D 点云。
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts.astype(np.float64))
    # 构建场景点云 KD 树。
    kd = o3d.geometry.KDTreeFlann(scene_pcd)
    # 记录命中半径邻域的点数。
    hit = 0
    for i in range(model_pts_transformed.shape[0]):
        # 搜索当前模型点在给定半径内是否存在场景邻居。
        k, _, _ = kd.search_radius_vector_3d(model_pts_transformed[i], radius)
        if k > 0:
            hit += 1
    # 返回命中比例。
    return float(hit) / max(1, model_pts_transformed.shape[0])


# 统一计算注册任务的若干评价指标。
def compute_metrics(
    model_pts: np.ndarray,
    scene_pts: np.ndarray,
    T_pred: np.ndarray,
    T_gt: Optional[np.ndarray] = None,
    inlier_radius: float = 5.0
) -> Dict[str, Any]:
    # 初始化输出字典。
    out: Dict[str, Any] = {}
    # 如果没有 GT 位姿，则 GT 相关指标全部置为 NaN。
    if T_gt is None:
        out["ADD"] = float("nan")
        out["ADD_S"] = float("nan")
        out["rotation_error_deg"] = float("nan")
        out["translation_error"] = float("nan")
    else:
        # 计算 ADD 指标。
        out["ADD"] = add_metric(model_pts, T_pred, T_gt)
        # 计算 ADD-S 指标。
        out["ADD_S"] = add_s_metric(model_pts, T_pred, T_gt)
        # 计算旋转和平移误差。
        rerr, terr = rotation_translation_error(T_pred, T_gt)
        out["rotation_error_deg"] = rerr
        out["translation_error"] = terr

    # 无论是否有 GT，都计算预测位姿下的内点比例。
    Pp = (T_pred[:3, :3] @ model_pts.T).T + T_pred[:3, 3]
    out["inlier_ratio"] = inlier_ratio_model_to_scene(Pp, scene_pts, radius=inlier_radius)
    # 返回指标字典。
    return out
