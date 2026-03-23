# preprocess.py
# 中文说明：
# 该文件负责点云预处理，包括：
# 1）固定 voxel 下采样 + 法向估计
# 2）模型法向方向修正
# 3）新增：按原始点数自适应控制下采样后的点数规模

# 导入拷贝模块，用于在不修改原始点云对象的前提下返回副本。
import copy

# 导入类型标注工具，保证函数接口清晰。
from typing import Dict, Any, Optional, Tuple

# 导入 numpy，用于数组与几何尺度计算。
import numpy as np

# 导入 open3d，用于点云下采样与法向估计。
import open3d as o3d


# 中文说明：
# 对场景点云执行“固定 voxel 大小”的下采样，并估计法向量。
# 这是你原有接口，保持不变，方便兼容已有调用代码。
def subsample_and_calculate_normals_scene(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 5.0,
    k: int = 5,
) -> o3d.geometry.PointCloud:
    # 若输入点云为空，则直接返回空点云副本。
    if len(pcd.points) == 0:
        return copy.deepcopy(pcd)

    # 使用体素下采样降低点云密度。
    cloud_down = pcd.voxel_down_sample(voxel_size)

    # 如果下采样后为空，则直接返回该空点云。
    if len(cloud_down.points) == 0:
        return cloud_down

    # 使用 KNN 方法估计法向量。
    cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

    # 返回处理后的场景点云。
    return cloud_down


# 中文说明：
# 对模型点云执行“固定 voxel 大小”的下采样、法向估计和法向翻转修正。
# 该函数保持了你原始代码的行为，接口也保持不变。
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

    # 若输入点云为空，则直接返回空点云副本。
    if len(pcd.points) == 0:
        return copy.deepcopy(pcd)

    # 先对模型点云做体素下采样。
    cloud_down = pcd.voxel_down_sample(voxel_size)

    # 若下采样后为空，则直接返回。
    if len(cloud_down.points) == 0:
        return cloud_down

    # 对下采样后的点云估计法向量。
    cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

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

    # 遍历每个法向量并执行方向修正。
    for i in range(len(normals)):
        # 优先使用原始点云中对应位置的点来决定法向量朝向。
        if i < len(orig_pts):
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid

        # 读取当前法向量。
        n = normals[i]

        # 计算法向量长度。
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

        # 写回修正后的法向量。
        normals[i] = n

    # 将修正后的法向量写回点云对象。
    cloud_down.normals = o3d.utility.Vector3dVector(normals)

    # 返回处理后的模型点云。
    return cloud_down


# 中文说明：
# 根据原始点数，决定是否需要下采样，以及下采样目标点数应设为多少。
# 返回：
# - None：表示不下采样
# - int：表示目标点数
def get_target_points_by_raw_count(
    n_raw: int,
    no_downsample_thresh: int = 450,
    mid_thresh: int = 1500,
    large_thresh: int = 3000,
    target_mid: int = 500,
    target_large: int = 450,
    target_xlarge: int = 400,
) -> Optional[int]:
    # 若点数已经较少，则不再下采样。
    if n_raw <= no_downsample_thresh:
        return None

    # 若点数处于中等规模，则压到较温和的目标点数。
    if n_raw <= mid_thresh:
        return target_mid

    # 若点数处于较大规模，则压到更紧凑的目标点数。
    if n_raw <= large_thresh:
        return target_large

    # 若点数非常大，则压到最紧凑目标点数。
    return target_xlarge


# 中文说明：
# 该函数通过二分搜索 voxel_size，使下采样后的点数尽量接近目标点数。
# 这是“控制结果（点数）”而不是“控制参数（固定 voxel）”的关键函数。
def _downsample_by_target_points(
    pcd: o3d.geometry.PointCloud,
    target_points: int,
    min_points_keep: int = 450,
    search_steps: int = 12,
    voxel_min: float = 1.0e-3,
    voxel_max_scale: float = 0.20,
) -> Tuple[o3d.geometry.PointCloud, float, int]:
    # 读取原始点数。
    n_raw = len(pcd.points)

    # 若为空点云，则直接返回副本和零参数。
    if n_raw == 0:
        return copy.deepcopy(pcd), 0.0, 0

    # 若原始点数本来就不多，则不下采样。
    if n_raw <= min_points_keep:
        return copy.deepcopy(pcd), 0.0, n_raw

    # 取出点坐标数组。
    pts = np.asarray(pcd.points)

    # 计算包围盒最小点。
    bbox_min = pts.min(axis=0)

    # 计算包围盒最大点。
    bbox_max = pts.max(axis=0)

    # 计算包围盒对角线长度，作为尺度估计。
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    # 设置搜索下界。
    voxel_low = voxel_min

    # 设置搜索上界。
    # 上界与几何尺度相关，这样不同实例大小下搜索范围更合理。
    voxel_high = max(voxel_min * 10.0, bbox_diag * voxel_max_scale)

    # 初始化“最佳结果”为原始点云。
    best_pcd = copy.deepcopy(pcd)

    # 初始化最佳 voxel。
    best_voxel = 0.0

    # 初始化最佳输出点数。
    best_n = n_raw

    # 初始化最佳误差。
    best_err = abs(n_raw - target_points)

    # 执行多轮二分搜索。
    for _ in range(search_steps):
        # 取当前中间 voxel。
        mid = 0.5 * (voxel_low + voxel_high)

        # 对当前 voxel 执行下采样。
        down = pcd.voxel_down_sample(mid)

        # 获取下采样后点数。
        n_down = len(down.points)

        # 计算与目标点数的绝对误差。
        err = abs(n_down - target_points)

        # 若当前结果更接近目标点数，且结果非空，则更新最佳解。
        if n_down > 0 and err < best_err:
            best_err = err
            best_pcd = down
            best_voxel = mid
            best_n = n_down

        # 若当前下采样后点数仍大于目标点数，说明 voxel 还不够大。
        if n_down > target_points:
            voxel_low = mid
        else:
            # 否则说明 voxel 偏大或刚好，应该往更小的方向收缩。
            voxel_high = mid

    # 返回最佳点云、最佳 voxel 和最佳输出点数。
    return best_pcd, best_voxel, best_n


# 中文说明：
# 对场景点云执行自适应下采样 + 法向估计。
# 返回：
# 1）处理后的点云
# 2）调试信息字典
def adaptive_subsample_and_calculate_normals_scene(
    pcd: o3d.geometry.PointCloud,
    k: int = 5,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    # 若未提供配置，则使用空字典。
    cfg = cfg or {}

    # 读取原始点数。
    n_raw = len(pcd.points)

    # 根据原始点数自动决定目标点数。
    target = get_target_points_by_raw_count(
        n_raw=n_raw,
        no_downsample_thresh=int(cfg.get("no_downsample_thresh", 450)),
        mid_thresh=int(cfg.get("mid_thresh", 1500)),
        large_thresh=int(cfg.get("large_thresh", 3000)),
        target_mid=int(cfg.get("target_mid", 500)),
        target_large=int(cfg.get("target_large", 450)),
        target_xlarge=int(cfg.get("target_xlarge", 400)),
    )

    # 若 target 为 None，说明该样本不需要下采样。
    if target is None:
        cloud_down = copy.deepcopy(pcd)
        voxel_used = 0.0
    else:
        # 否则通过二分搜索自动寻找合适的 voxel。
        cloud_down, voxel_used, _ = _downsample_by_target_points(
            pcd=pcd,
            target_points=target,
            min_points_keep=int(cfg.get("no_downsample_thresh", 450)),
            search_steps=int(cfg.get("search_steps", 12)),
            voxel_min=float(cfg.get("voxel_min", 1.0e-3)),
            voxel_max_scale=float(cfg.get("voxel_max_scale", 0.20)),
        )

    # 若结果为空点云，则直接返回。
    if len(cloud_down.points) == 0:
        return cloud_down, {
            "raw_points": n_raw,
            "down_points": 0,
            "target_points": target,
            "voxel_used": voxel_used,
        }

    # 对结果点云估计法向量。
    cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

    # 返回点云和调试信息。
    return cloud_down, {
        "raw_points": n_raw,
        "down_points": len(cloud_down.points),
        "target_points": target,
        "voxel_used": voxel_used,
    }


# 中文说明：
# 对模型点云执行自适应下采样 + 法向估计 + 法向方向修正。
# 注意：
# - 该函数主要为“将来 model 端也自适应”预留
# - 默认配置下你可以先不启用它，以免影响模型缓存
def adaptive_subsample_and_calculate_normals_model(
    pcd: o3d.geometry.PointCloud,
    k: int = 5,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    # 若未提供配置，则使用空字典。
    cfg = cfg or {}

    # 读取原始点数。
    n_raw = len(pcd.points)

    # 根据原始点数自动决定目标点数。
    target = get_target_points_by_raw_count(
        n_raw=n_raw,
        no_downsample_thresh=int(cfg.get("no_downsample_thresh", 450)),
        mid_thresh=int(cfg.get("mid_thresh", 1500)),
        large_thresh=int(cfg.get("large_thresh", 3000)),
        target_mid=int(cfg.get("target_mid", 500)),
        target_large=int(cfg.get("target_large", 450)),
        target_xlarge=int(cfg.get("target_xlarge", 400)),
    )

    # 若无需下采样，则直接复制原始点云。
    if target is None:
        cloud_down = copy.deepcopy(pcd)
        voxel_used = 0.0
    else:
        # 否则自动搜索适配该样本的 voxel 大小。
        cloud_down, voxel_used, _ = _downsample_by_target_points(
            pcd=pcd,
            target_points=target,
            min_points_keep=int(cfg.get("no_downsample_thresh", 450)),
            search_steps=int(cfg.get("search_steps", 12)),
            voxel_min=float(cfg.get("voxel_min", 1.0e-3)),
            voxel_max_scale=float(cfg.get("voxel_max_scale", 0.20)),
        )

    # 若下采样结果为空，则直接返回。
    if len(cloud_down.points) == 0:
        return cloud_down, {
            "raw_points": n_raw,
            "down_points": 0,
            "target_points": target,
            "voxel_used": voxel_used,
        }

    # 对下采样后点云估计法向量。
    cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

    # 取出下采样点。
    pts_down = np.asarray(cloud_down.points)

    # 若点云为空，则直接返回。
    if pts_down.shape[0] == 0:
        return cloud_down, {
            "raw_points": n_raw,
            "down_points": 0,
            "target_points": target,
            "voxel_used": voxel_used,
        }

    # 计算下采样后点云质心。
    centroid = pts_down.mean(axis=0)

    # 拷贝法向量数组。
    normals = np.asarray(cloud_down.normals).copy()

    # 读取原始点云点坐标。
    orig_pts = np.asarray(pcd.points)

    # 对每个法向量执行与原有 model 预处理一致的方向修正逻辑。
    for i in range(len(normals)):
        # 若原始点云中有对应索引点，则优先参考该点。
        if i < len(orig_pts):
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid

        # 读取当前法向。
        n = normals[i]

        # 计算长度。
        n_norm = np.linalg.norm(n)

        # 若为零法向，则退化为参考方向或默认 z 轴。
        if n_norm == 0.0:
            n = orientation_reference
            n_norm = np.linalg.norm(n)

            if n_norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
        else:
            # 若法向与参考方向同向，则翻转。
            if float(n @ orientation_reference) > 0.0:
                n = -n

        # 写回法向。
        normals[i] = n

    # 将修正后的法向写回点云对象。
    cloud_down.normals = o3d.utility.Vector3dVector(normals)

    # 返回结果与调试信息。
    return cloud_down, {
        "raw_points": n_raw,
        "down_points": len(cloud_down.points),
        "target_points": target,
        "voxel_used": voxel_used,
    }