from typing import Optional, Tuple
import numpy as np
import open3d as o3d


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
    reg = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return reg.transformation
