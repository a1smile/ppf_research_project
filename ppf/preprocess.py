import numpy as np
import open3d as o3d


def subsample_and_calculate_normals_scene(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 5.0,
    k: int = 5,
) -> o3d.geometry.PointCloud:
    cloud_down = pcd.voxel_down_sample(voxel_size)
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return cloud_down


def subsample_and_calculate_normals_model(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 5.0,
    k: int = 5,
) -> o3d.geometry.PointCloud:
    """
    Strictly mirrors the baseline quirk:
    - flip normals toward viewpoint=centroid of downsampled cloud
    - but references ORIGINAL cloud's i-th point when deciding orientation
    """
    cloud_down = pcd.voxel_down_sample(voxel_size)
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    pts_down = np.asarray(cloud_down.points)
    if pts_down.shape[0] == 0:
        return cloud_down

    centroid = pts_down.mean(axis=0)
    normals = np.asarray(cloud_down.normals).copy()
    orig_pts = np.asarray(pcd.points)

    for i in range(len(normals)):
        if i < len(orig_pts):
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid

        n = normals[i]
        n_norm = np.linalg.norm(n)
        if n_norm == 0.0:
            n = orientation_reference
            n_norm = np.linalg.norm(n)
            if n_norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
        else:
            if float(n @ orientation_reference) > 0.0:
                n = -n
        normals[i] = n

    cloud_down.normals = o3d.utility.Vector3dVector(normals)
    return cloud_down
