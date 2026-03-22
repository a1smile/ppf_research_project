# registration.py：在尽量保持原有结构的前提下，负责完整的 PPF 注册流程。

# 导入数学库，用于角度、弧度和三角函数计算。
import math
# 导入时间库，用于统计流程耗时。
import time
# 导入日志库，用于输出运行信息。
import logging
# 导入数据类装饰器，用于定义结构化结果。
from dataclasses import dataclass
# 导入类型标注工具，便于描述接口。
from typing import Dict, Any, Optional, Tuple, List

# 中文说明：该模块实现从模型和场景点云到最终位姿估计的完整注册流程。

# 导入 NumPy，用于数组和矩阵运算。
import numpy as np
# 导入 Open3D，用于点云读写、KDTree 和几何处理。
import open3d as o3d

# 导入通用工具函数，包括随机种子、计时器和刚体变换操作。
from .utils import (
    set_global_seed, Timer, make_affine, invert_affine, compose_affine,
    rotation_matrix_from_axis_angle, wrap_to_pi, ensure_dir
)
# 导入 PPF 特征计算与角度转换函数。
from .ppf_features import compute_pair_features, angle_from_transformed_point, to_internal_feature_g
# 导入 PPF 模型构建函数和模型类型。
from .model_builder import build_ppf_model, PPFModel
# 导入模型与场景点云的下采样和法向估计函数。
from .preprocess import subsample_and_calculate_normals_model, subsample_and_calculate_normals_scene
# 导入原始姿态聚类结构和聚类函数。
from .clustering import PoseWithVotes, cluster_poses
# 导入鲁棒投票器和其统计结构。
from .voting_robust import RobustVoter, RobustVoteStats
# 导入 KDE 均值漂移精修器。
from .kde_refine import KDEMeanShiftRefiner
# 导入点到点 ICP 精修函数。
from .refine_icp import refine_icp_point_to_point

# 导入模型缓存加载函数，用于直接读取预构建的 PPF 模型。
from .model_cache_io import load_ppf_model_cache

# 导入姿态候选转换和姿态聚类函数。
from .pose_clustering import hypotheses_from_posewithvotes, cluster_pose_hypotheses


# 计算场景参考点到参考坐标系的刚体变换。
def compute_transform_sg(scene_ref_p: np.ndarray, scene_ref_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 定义目标对齐的 x 轴和退化时备用的 y 轴。
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)
    # 对参考点法向量做归一化。
    n = scene_ref_n / (np.linalg.norm(scene_ref_n) + 1e-12)
    # 通过叉积计算将法向量对齐到 x 轴的旋转轴。
    axis = np.cross(n, ex)
    # 计算旋转轴的模长。
    axis_norm = np.linalg.norm(axis)
    # 若旋转轴长度为 0，说明法向量已与 x 轴平行，改用 y 轴作为退化轴。
    if axis_norm == 0.0:
        axis = ey.copy()
        axis_norm = 1.0
    # 将旋转轴归一化。
    axis /= axis_norm
    # 根据点积计算法向量和 x 轴之间的夹角。
    angle = math.acos(max(-1.0, min(1.0, float(n @ ex))))
    # 根据旋转轴和角度生成旋转矩阵。
    R = rotation_matrix_from_axis_angle(axis, angle)
    # 计算平移项，使参考点被变换到局部坐标原点。
    t = R @ (-scene_ref_p)
    # 拼装 4x4 仿射变换矩阵。
    T = make_affine(R, t)
    # 返回完整变换和旋转矩阵。
    return T, R


# 定义注册流程的统计信息结构。
@dataclass
class RegistrationStats:
    # 模型构建或缓存加载耗时。
    model_build_time: float
    # 注册核心流程耗时。
    registration_time: float
    # 总耗时。
    total_time: float
    # 候选膨胀倍数均值。
    candidate_inflation_mean: float
    # 鲁棒投票统计摘要。
    robust_vote_summary: dict
    # KDE 精修调用次数。
    kde_refine_calls: int


# 执行 PPF 注册主循环，并返回最终位姿和调试信息。
def ppf_register(
    model: PPFModel,
    scene_pcd: o3d.geometry.PointCloud,
    cfg: dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Baseline-compatible core loop with optional enhancements.
    Returns (T_final, debug_info).
    """
    # 读取三种增强模块的开关状态。
    enable_rsmrq = bool(cfg.get("enable_rsmrq", False))
    enable_robust = bool(cfg.get("enable_robust_vote", False))
    enable_kde = bool(cfg.get("enable_kde_refine", False))

    # 读取姿态聚类配置。
    pose_cluster_cfg = cfg.get("pose_clustering", {})
    enable_pose_clustering = bool(pose_cluster_cfg.get("enable", True))
    pose_cluster_pos_thresh = float(pose_cluster_cfg.get("pos_thresh", 20.0))
    pose_cluster_rot_thresh_deg = float(pose_cluster_cfg.get("rot_thresh_deg", 20.0))
    # 将姿态聚类角度阈值从度转换为弧度。
    pose_cluster_rot_thresh_rad = math.radians(pose_cluster_rot_thresh_deg)
    pose_cluster_min_size = int(pose_cluster_cfg.get("min_cluster_size", 1))
    pose_cluster_merge_by_score = bool(pose_cluster_cfg.get("merge_by_score", True))

    # 按配置按需构造鲁棒投票器和 KDE 精修器。
    voter = RobustVoter(cfg.get("robust_vote", {}), logger=logger) if enable_robust else None
    refiner = KDEMeanShiftRefiner(cfg.get("kde_refine", {}), logger=logger) if enable_kde else None

    # 读取场景点和法向量数组。
    scene_pts = np.asarray(scene_pcd.points).astype(np.float64)
    scene_normals = np.asarray(scene_pcd.normals).astype(np.float64)
    # 记录场景点总数。
    Nscene = len(scene_pts)

    # 根据角度步长计算投票累加器的列数。
    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    # 鲁棒投票使用浮点累加器，否则使用整型累加器。
    acc_dtype = np.float32 if enable_robust else np.int32
    # 初始化二维投票累加器。
    accumulator = np.zeros((len(model.pts), aux_size), dtype=acc_dtype)

    # KDE 样本缓存，键为 (mr, bin)，值为 (alpha, weight) 列表。
    samples = {}  # type: ignore

    # 为场景点云构建 KDTree。
    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    # 邻域搜索半径取模型直径的一半。
    radius = float(model.model_diameter) / 2.0

    # 保存投票得到的姿态候选。
    voted_poses: List[PoseWithVotes] = []

    # 保存候选膨胀统计。
    candidate_inflations = []
    # 初始化鲁棒投票统计。
    rv_stats = RobustVoteStats()
    # 记录 KDE 精修调用次数。
    kde_calls = 0

    # 读取场景参考点采样步长。
    scene_ref_sampling_rate = int(cfg.get("scene_ref_sampling_rate", 20))
    # 读取原始聚类使用的平移阈值。
    pos_thresh = float(cfg.get("pos_thresh", 0.005))
    # 读取原始聚类使用的旋转阈值并转换为弧度。
    rot_thresh_rad = math.radians(float(cfg.get("rot_thresh_deg", 30.0)))

    # 按设定步长遍历场景参考点。
    for sr in range(0, Nscene, scene_ref_sampling_rate):
        # 取出当前场景参考点和其法向量。
        sr_p = scene_pts[sr]
        sr_n = scene_normals[sr]
        # 计算当前参考点对应的局部坐标变换。
        T_sg, R_sg = compute_transform_sg(sr_p, sr_n)

        # 在搜索半径内查找邻域点。
        _, idxs, _ = kdtree.search_radius_vector_3d(scene_pcd.points[sr], radius)

        # 遍历邻域点，与当前参考点组成点对。
        for si in idxs:
            # 跳过参考点自身。
            if si == sr:
                continue
            # 取出邻域点坐标和法向量。
            si_p = scene_pts[si]
            si_n = scene_normals[si]

            # 计算当前场景点对的 PPF 特征。
            feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
            # 若特征无效则跳过。
            if feat is None:
                continue
            # 拆包四维特征。
            f1, f2, f3, f4 = feat
            # 转换为哈希查询内部格式。
            gs = to_internal_feature_g(f1, f2, f3, f4)

            # 将邻域点变换到参考点局部坐标系。
            si_trans = R_sg @ (si_p - sr_p)
            # 计算局部角度。
            ang = angle_from_transformed_point(si_trans)
            # 场景侧角度项取负号。
            alpha_s = -ang

            # 根据特征查询模型哈希桶。
            buckets = model.hash_table.query_buckets(gs)
            # 统计候选膨胀倍数，比较总候选量与单桶候选量。
            base_sz = len(buckets[0]) if len(buckets) > 0 else 0
            tot_sz = sum(len(b) for b in buckets)
            infl = (tot_sz / max(1, base_sz)) if base_sz > 0 else (float(tot_sz) if tot_sz > 0 else 1.0)
            candidate_inflations.append(float(infl))

            # 合并候选，键为 (模型参考点索引, 模型邻点索引)。
            merged = {}  # (mr,mi)->(entry,count)
            if enable_rsmrq:
                # RSMRQ 模式下允许对每个桶分别做 top-m 截断后再合并。
                for b in buckets:
                    # 空桶直接跳过。
                    if not b:
                        continue
                    if enable_robust:
                        # 收集候选中存储的模型特征。
                        gm_list = []
                        for e in b:
                            # 若缺失特征则跳过该候选。
                            if e.g is None:
                                continue
                            gm_list.append(np.array(e.g, dtype=np.float32))
                        # 如果有缺失特征，这里保持原逻辑不处理。
                        if len(gm_list) != len(b):
                            pass
                        # 将特征堆叠成数组，便于批量计算残差。
                        gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)
                        # 若没有有效特征则跳过该桶。
                        if gm_arr.shape[0] == 0:
                            continue
                        # 计算场景特征和桶内候选特征之间的归一化残差。
                        res = np.linalg.norm((gm_arr - gs[None, :]) / np.array(
                            [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)], dtype=np.float32
                        ), axis=1)
                        # 仅保留残差最小的 top-m 候选。
                        idx_keep = voter.select_top_m(res)  # type: ignore
                        # 更新截断统计。
                        rv_stats.update_trunc(int(idx_keep.shape[0]), int(res.shape[0]))
                        for ii in idx_keep:
                            # 取出保留候选。
                            e = b[int(ii)]
                            # 使用模型索引对作为合并键。
                            key = (e.mr, e.mi)
                            if key not in merged:
                                merged[key] = (e, 1)
                            else:
                                merged[key] = (merged[key][0], merged[key][1] + 1)
                    else:
                        # 非鲁棒模式下，不做 top-m 截断，直接合并所有候选。
                        for e in b:
                            key = (e.mr, e.mi)
                            if key not in merged:
                                merged[key] = (e, 1)
                            else:
                                merged[key] = (merged[key][0], merged[key][1] + 1)

                # union 模式下只保留唯一候选，不累计重复次数。
                if model.merge_mode == "union":
                    merged = {k: (v[0], 1) for k, v in merged.items()}

            else:
                # 基线模式下只使用第一个桶。
                b = buckets[0] if buckets else []
                if enable_robust and b:
                    # 收集带有模型特征的候选。
                    gm_list = [np.array(e.g, dtype=np.float32) for e in b if e.g is not None]
                    # 堆叠成数组，便于批量计算。
                    gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)
                    if gm_arr.shape[0] > 0:
                        # 计算场景特征与候选特征之间的残差。
                        res = np.linalg.norm((gm_arr - gs[None, :]) / np.array(
                            [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)], dtype=np.float32
                        ), axis=1)
                        # 保留 top-m 候选。
                        idx_keep = voter.select_top_m(res)  # type: ignore
                        # 更新截断统计。
                        rv_stats.update_trunc(int(idx_keep.shape[0]), int(res.shape[0]))
                        for ii in idx_keep:
                            # 将保留候选按唯一键写入。
                            e = b[int(ii)]
                            merged[(e.mr, e.mi)] = (e, 1)
                    else:
                        pass
                else:
                    # 普通基线模式下，直接保留桶内全部候选。
                    for e in b:
                        merged[(e.mr, e.mi)] = (e, 1)

            # 对合并后的候选执行投票。
            for (mr, mi), (e, cnt) in merged.items():
                # 计算模型侧角度和场景侧角度的差值。
                alpha = model.alpha_m[mr][mi] - alpha_s
                # 将角度规范化到 [-pi, pi] 区间。
                alpha = wrap_to_pi(alpha)
                # 将角度离散到投票 bin。
                bin_j = int(math.floor((alpha + math.pi) / model.angle_step))
                bin_j = max(0, min(aux_size - 1, bin_j))

                if not enable_robust:
                    # 非鲁棒模式下按计数或并集方式写入累加器。
                    accumulator[mr, bin_j] += int(cnt) if model.merge_mode == "count" else 1
                    if enable_kde:
                        # 若启用 KDE，则记录样本角度和统一权重。
                        key = (mr, bin_j)
                        samples.setdefault(key, []).append((alpha, 1.0))
                    continue

                # 鲁棒模式需要模型特征计算残差。
                if e.g is None:
                    continue
                # 将模型特征转为数组。
                gm = np.array(e.g, dtype=np.float32)
                # 计算残差。
                r = voter.residual(gs, gm, model.model_diameter)  # type: ignore
                # 根据残差计算鲁棒权重。
                w = voter.compute_weight(r)  # type: ignore
                # count 模式下，将重复次数乘到权重上。
                if model.merge_mode == "count":
                    w *= float(cnt)
                # 将权重投票写入累加器。
                voter.vote(accumulator, mr, bin_j, w, rv_stats)  # type: ignore

                if enable_kde:
                    # 记录用于 KDE 精修的样本角度及其权重。
                    key = (mr, bin_j)
                    samples.setdefault(key, []).append((alpha, float(w)))

        # 每个场景参考点处理完后，读取当前累加器全局最大值。
        row_max = float(np.max(accumulator)) if accumulator.size > 0 else 0.0
        if logger:
            # 输出当前参考点处理后的累加器峰值和膨胀均值。
            logger.info(
                f"[SR {sr}] accumulator_global_max={row_max:.3f} "
                f"candidate_inflation_mean_sofar={float(np.mean(candidate_inflations)):.3f}"
            )

        # 对每个模型参考点读取最佳 bin，并生成姿态候选。
        for mr in range(len(model.pts)):
            # 取出当前参考点对应的一行累加器。
            row = accumulator[mr]
            # 最大票数作为当前姿态分数。
            votes = float(np.max(row))
            # 若票数不大于 0，则跳过。
            if votes <= 0:
                continue
            # 取出峰值 bin 索引。
            bj = int(np.argmax(row))
            # 用 bin 中心恢复初始角度。
            theta0 = (float(bj) + 0.5) * model.angle_step - math.pi

            # 默认直接使用离散角度。
            theta = theta0
            if enable_kde and refiner is not None:
                # 取当前参考点 top-k 个高票 bin 做 KDE 精修。
                k = min(refiner.top_k, aux_size)
                if k > 1:
                    # 找出票数最高的前 k 个 bin。
                    top_idx = np.argpartition(-row, k - 1)[:k]
                    # 汇总这些 bin 下缓存的样本角度与权重。
                    ts = []
                    ws = []
                    for b in top_idx:
                        key = (mr, int(b))
                        if key in samples:
                            for a, w in samples[key]:
                                ts.append(a)
                                ws.append(w)
                    # 样本足够时执行 KDE 精修。
                    if len(ts) >= 3:
                        theta, trace = refiner.refine(
                            np.array(ts, dtype=np.float32),
                            np.array(ws, dtype=np.float32),
                            theta_init=theta0
                        )
                        kde_calls += 1
                        if logger:
                            # 输出精修前后的角度和迭代次数。
                            logger.info(
                                f"[KDERefine] mr={mr} theta0={theta0:.4f} "
                                f"theta={theta:.4f} iters={trace.iters}"
                            )

            # 构造绕 x 轴的旋转矩阵。
            R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), float(theta))
            # 构造仅含旋转的仿射矩阵。
            T_x = make_affine(R_x, np.zeros(3, dtype=float))

            # 构造模型参考点到模型局部坐标系的变换。
            T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
            # 组合场景局部坐标、旋转校正和模型局部变换，得到完整姿态。
            T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
            # 保存该姿态候选和票数。
            voted_poses.append(PoseWithVotes(T, votes))

        # 清空累加器，为下一个参考点重新投票。
        accumulator.fill(0)
        # 清空 KDE 样本缓存。
        samples.clear()

    # 先尝试新的姿态聚类，再回退到原始 cluster_poses。
    pose_cluster_debug = {}
    T_final = None

    # 若启用姿态聚类且已有候选位姿，则优先执行新聚类。
    if enable_pose_clustering and len(voted_poses) > 0:
        # 将 PoseWithVotes 转为姿态聚类模块的输入格式。
        hypos = hypotheses_from_posewithvotes(voted_poses)
        # 对所有姿态候选进行聚类。
        best_cluster, clusters_pc, pose_cluster_debug = cluster_pose_hypotheses(
            hypos,
            pos_thresh=pose_cluster_pos_thresh,
            rot_thresh_rad=pose_cluster_rot_thresh_rad,
            min_cluster_size=pose_cluster_min_size,
            merge_by_score=pose_cluster_merge_by_score,
        )

        if logger:
            # 输出姿态聚类的摘要统计。
            logger.info(
                "[PoseClustering] "
                f"enabled={enable_pose_clustering} "
                f"num_hypotheses={pose_cluster_debug.get('num_hypotheses', 0)} "
                f"num_clusters={pose_cluster_debug.get('num_clusters', 0)} "
                f"best_cluster_size={pose_cluster_debug.get('best_cluster_size', 0)} "
                f"best_cluster_score_sum={pose_cluster_debug.get('best_cluster_score_sum', 0.0):.3f}"
            )

        # 若找到最佳簇，则使用其代表位姿。
        if best_cluster is not None:
            T_final = best_cluster.T_rep

    # 若姿态聚类未得到结果，则回退到原始聚类方法。
    if T_final is None:
        best = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
        # 若原始聚类也未得到结果，则退化为单位矩阵。
        T_final = best[0].T if best else np.eye(4, dtype=float)

    # 汇总调试信息。
    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "robust_vote": rv_stats.summary() if enable_robust else {},
        "kde_refine_calls": kde_calls,
        "pose_clustering": pose_cluster_debug,
    }
    # 返回最终位姿和调试信息。
    return T_final, debug


# 对外暴露的完整注册入口，负责读取点云、构建或加载模型并执行注册。
def run_registration(
    model_path: str,
    scene_path: str,
    cfg: dict,
    logger: Optional[logging.Logger] = None,
    # 可选模型缓存路径，提供后将不再重新构建 PPF 模型。
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    """
    Paper-grade entry point:
    - preprocess model/scene (baseline-compatible)
    - build model hash（或加载缓存）
    - run registration (with optional enhancements)
    - optional ICP refinement
    """
    # 读取随机种子并设置全局随机状态。
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    # 读取基础参数，并计算角度步长和距离步长。
    sampling_leaf = float(cfg.get("sampling_leaf", 5.0))
    normal_k = int(cfg.get("normal_k", 5))
    angle_step_deg = float(cfg.get("angle_step_deg", 12.0))
    # 将角度步长转换为弧度。
    angle_step = math.radians(angle_step_deg)
    # 按采样尺度比例计算距离离散步长。
    distance_step = float(cfg.get("distance_step_ratio", 0.6)) * sampling_leaf

    # 记录总流程起始时间。
    t0 = time.perf_counter()

    # 读取模型点云。
    cloud_model = o3d.io.read_point_cloud(model_path)
    # 若模型点云为空，则直接报错。
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")
    # 读取场景点云。
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    # 若场景点云为空，则直接报错。
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Empty scene: {scene_path}")

    # 初始化下采样后的模型点云变量。
    model_down = None
    if model_cache_path:
        # 若提供缓存路径，则优先加载缓存模型。
        with Timer("model_load_cache", logger=logger) as tm:
            ppf_model = load_ppf_model_cache(model_cache_path, cfg=cfg, strict=strict_cache)
    else:
        # 否则先对模型点云下采样并计算法向，再构建 PPF 模型。
        with Timer("model_build", logger=logger) as tm:
            model_down = subsample_and_calculate_normals_model(cloud_model, voxel_size=sampling_leaf, k=normal_k)
            ppf_model = build_ppf_model(model_down, angle_step, distance_step, cfg, logger=logger)

    # 对场景点云做预处理并执行注册。
    with Timer("registration", logger=logger) as tr:
        scene_down = subsample_and_calculate_normals_scene(cloud_scene, voxel_size=sampling_leaf, k=normal_k)
        T_pred, debug = ppf_register(ppf_model, scene_down, cfg, logger=logger)

    # 按需在下采样点云上执行 ICP 精修。
    icp_cfg = cfg.get("icp_refine", {})
    if bool(icp_cfg.get("enable", False)):
        if logger:
            logger.info("[ICP] enabled, refining...")
        # 缓存模式下这里补做一次模型下采样，供 ICP 使用。
        if model_down is None:
            model_down = subsample_and_calculate_normals_model(cloud_model, voxel_size=sampling_leaf, k=normal_k)
        # 以当前预测位姿为初值做点到点 ICP。
        T_ref = refine_icp_point_to_point(
            source=model_down,
            target=scene_down,
            init_T=T_pred,
            distance_threshold=float(icp_cfg.get("distance_threshold", 5.0)),
            max_iter=int(icp_cfg.get("max_iter", 30))
        )
        # 用 ICP 结果覆盖预测位姿。
        T_pred = T_ref

    # 构造输出模型点云，并将模型点变换到场景坐标系。
    out_model = o3d.geometry.PointCloud()
    # 对模型点做刚体变换。
    P = (T_pred[:3, :3] @ ppf_model.pts.T).T + T_pred[:3, 3]
    # 将变换后的点写回点云对象。
    out_model.points = o3d.utility.Vector3dVector(P)

    # 计算总耗时。
    total_time = time.perf_counter() - t0

    # 组织统计结果。
    stats = RegistrationStats(
        model_build_time=tm.elapsed,
        registration_time=tr.elapsed,
        total_time=total_time,
        candidate_inflation_mean=float(debug.get("candidate_inflation_mean", 1.0)),
        robust_vote_summary=debug.get("robust_vote", {}),
        kde_refine_calls=int(debug.get("kde_refine_calls", 0))
    )

    # 返回预测位姿、变换后的模型点云、调试信息和统计信息。
    return T_pred, out_model, debug, stats
