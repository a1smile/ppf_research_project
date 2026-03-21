# 导入 numpy 和 open3d。
import numpy as np
import open3d as o3d


# 对场景点云进行下采样并估计法向量。
def subsample_and_calculate_normals_scene(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 5.0,
    k: int = 5,
) -> o3d.geometry.PointCloud:
    # 使用体素下采样降低场景点云密度。
    cloud_down = pcd.voxel_down_sample(voxel_size)
    # 使用 KNN 方式估计下采样后点云的法向量。
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    # 返回处理后的场景点云。
    return cloud_down


# 对模型点云进行下采样、法向量估计和方向修正。
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
    # 先对模型点云做体素下采样。
    cloud_down = pcd.voxel_down_sample(voxel_size)
    # 对下采样后的点云估计法向量。
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    # 取出下采样后点坐标。
    pts_down = np.asarray(cloud_down.points)
    # 若点云为空，则直接返回。
    if pts_down.shape[0] == 0:
        return cloud_down

    # 计算下采样点云的中心点。
    centroid = pts_down.mean(axis=0)
    # 复制法向量数组，后续会做原地修正。
    normals = np.asarray(cloud_down.normals).copy()
    # 读取原始点云坐标。
    orig_pts = np.asarray(pcd.points)

    for i in range(len(normals)):
        # 优先使用原始点云中对应位置的点来决定法向量朝向。
        if i < len(orig_pts):
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid

        # 读取当前法向量。
        n = normals[i]
        n_norm = np.linalg.norm(n)
        # 若当前法向量为零向量，则用参考方向代替。
        if n_norm == 0.0:
            n = orientation_reference
            n_norm = np.linalg.norm(n)
            # 若参考方向也为零，则退化为默认 z 轴方向。
            if n_norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
        else:
            # 若法向量与参考方向夹角为锐角，则翻转方向。
            if float(n @ orientation_reference) > 0.0:
                n = -n
        normals[i] = n

    # 将修正后的法向量写回点云对象。
    cloud_down.normals = o3d.utility.Vector3dVector(normals)
    # 返回处理后的模型点云。
    return cloud_down
