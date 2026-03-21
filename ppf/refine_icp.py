# 导入类型标注工具。
from typing import Optional, Tuple
# 导入 numpy 和 open3d。
import numpy as np
import open3d as o3d


# 使用点到点 ICP 对初始位姿进行进一步精化。
def refine_icp_point_to_point(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    distance_threshold: float = 5.0,
    max_iter: int = 30
) -> np.ndarray:
    """
    Optional ICP refinement (point-to-point).
    Returns refined 4x4 transform.
    """
    # 调用 Open3D 的 ICP 配准接口。
    reg = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    # 返回 ICP 优化后的变换矩阵。
    return reg.transformation
