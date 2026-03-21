# registration.py（最小改动版：尽量保持你原文件结构/行文不变，只加“缓存加载模型”分支）
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

# 中文说明：该模块实现从模型和场景点云到最终位姿估计的完整注册流程。

import numpy as np
import open3d as o3d

from .utils import (
    set_global_seed, Timer, make_affine, invert_affine, compose_affine,
    rotation_matrix_from_axis_angle, wrap_to_pi, ensure_dir
)
from .ppf_features import compute_pair_features, angle_from_transformed_point, to_internal_feature_g
from .model_builder import build_ppf_model, PPFModel
from .preprocess import subsample_and_calculate_normals_model, subsample_and_calculate_normals_scene
from .clustering import PoseWithVotes, cluster_poses
from .voting_robust import RobustVoter, RobustVoteStats
from .kde_refine import KDEMeanShiftRefiner
from .refine_icp import refine_icp_point_to_point

# ✅ 最小新增：导入缓存加载（你需要新建 model_cache_io.py，里面提供 load_ppf_model_cache）
from .model_cache_io import load_ppf_model_cache


# 计算场景参考点到参考坐标系的刚体变换。
def compute_transform_sg(scene_ref_p: np.ndarray, scene_ref_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 定义基准 x 轴和备用 y 轴。
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)
    # 对场景参考法向量做归一化。
    n = scene_ref_n / (np.linalg.norm(scene_ref_n) + 1e-12)
    # 计算将法向量对齐到 x 轴的旋转轴。
    axis = np.cross(n, ex)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        axis = ey.copy()
        axis_norm = 1.0
    axis /= axis_norm
    # 计算旋转角度。
    angle = math.acos(max(-1.0, min(1.0, float(n @ ex))))
    # 构造旋转矩阵。
    R = rotation_matrix_from_axis_angle(axis, angle)
    # 计算平移项，使参考点平移到局部原点。
    t = R @ (-scene_ref_p)
    # 组装为 4x4 仿射变换矩阵。
    T = make_affine(R, t)
    # 返回完整变换和旋转部分。
    return T, R


# 定义注册过程的统计结果结构。
@dataclass
class RegistrationStats:
    # 模型构建或缓存加载阶段耗时。
    model_build_time: float
    # 注册主循环耗时。
    registration_time: float
    # 整体流程总耗时。
    total_time: float
    # 候选膨胀的平均倍数。
    candidate_inflation_mean: float
    # 鲁棒投票模块统计摘要。
    robust_vote_summary: dict
    # KDE 精化调用次数。
    kde_refine_calls: int


# 执行 PPF 注册主循环，并返回最终位姿和调试信息。
def ppf_register(
    model: PPFModel,
    scene_pcd: o3d.geometry.PointCloud,
    cfg: dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Baseline-compatible core loop with 3 optional enhancements.
    Returns (T_final, debug_info).
    """
    # 读取三个增强模块的开关状态。
    enable_rsmrq = bool(cfg.get("enable_rsmrq", False))
    enable_robust = bool(cfg.get("enable_robust_vote", False))
    enable_kde = bool(cfg.get("enable_kde_refine", False))

    # 根据配置按需构造鲁棒投票器和 KDE 精化器。
    voter = RobustVoter(cfg.get("robust_vote", {}), logger=logger) if enable_robust else None
    refiner = KDEMeanShiftRefiner(cfg.get("kde_refine", {}), logger=logger) if enable_kde else None

    # 读取场景点和法向量数组。
    scene_pts = np.asarray(scene_pcd.points).astype(np.float64)
    scene_normals = np.asarray(scene_pcd.normals).astype(np.float64)
    Nscene = len(scene_pts)

    # 根据角度步长确定投票累加器的 bin 数量。
    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    # 鲁棒投票时使用浮点累加器，否则使用整型累加器。
    acc_dtype = np.float32 if enable_robust else np.int32
    accumulator = np.zeros((len(model.pts), aux_size), dtype=acc_dtype)

    # KDE samples: (mr, bin) -> list[(alpha, weight)]
    samples = {}  # type: ignore

    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    radius = float(model.model_diameter) / 2.0

    voted_poses: List[PoseWithVotes] = []

    candidate_inflations = []
    rv_stats = RobustVoteStats()
    kde_calls = 0

    scene_ref_sampling_rate = int(cfg.get("scene_ref_sampling_rate", 20))
    pos_thresh = float(cfg.get("pos_thresh", 0.005))
    rot_thresh_rad = math.radians(float(cfg.get("rot_thresh_deg", 30.0)))

    # 按设定步长遍历场景参考点。
    for sr in range(0, Nscene, scene_ref_sampling_rate):
        sr_p = scene_pts[sr]
        sr_n = scene_normals[sr]
        T_sg, R_sg = compute_transform_sg(sr_p, sr_n)

        # radius neighbors
        _, idxs, _ = kdtree.search_radius_vector_3d(scene_pcd.points[sr], radius)

        # iterate neighbor pairs
        for si in idxs:
            if si == sr:
                continue
            si_p = scene_pts[si]
            si_n = scene_normals[si]

            feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
            if feat is None:
                continue
            f1, f2, f3, f4 = feat
            gs = to_internal_feature_g(f1, f2, f3, f4)

            # alpha_s
            si_trans = R_sg @ (si_p - sr_p)
            ang = angle_from_transformed_point(si_trans)
            alpha_s = -ang

            # query buckets
            buckets = model.hash_table.query_buckets(gs)
            # candidate inflation statistic: compare total bucket sizes vs baseline-like (first bucket)
            base_sz = len(buckets[0]) if len(buckets) > 0 else 0
            tot_sz = sum(len(b) for b in buckets)
            infl = (tot_sz / max(1, base_sz)) if base_sz > 0 else (float(tot_sz) if tot_sz > 0 else 1.0)
            candidate_inflations.append(float(infl))

            # apply top-m per bucket (only meaningful for robust vote)
            merged = {}  # (mr,mi)->(entry,count)
            if enable_rsmrq:
                # manual merge to allow top-m truncation per bucket
                for b in buckets:
                    if not b:
                        continue
                    if enable_robust:
                        # require feature stored
                        gm_list = []
                        for e in b:
                            if e.g is None:
                                continue
                            gm_list.append(np.array(e.g, dtype=np.float32))
                        if len(gm_list) != len(b):
                            # if some missing (should not happen when robust enabled)
                            pass
                        gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)
                        if gm_arr.shape[0] == 0:
                            continue
                        res = np.linalg.norm((gm_arr - gs[None, :]) / np.array(
                            [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)], dtype=np.float32
                        ), axis=1)
                        idx_keep = voter.select_top_m(res)  # type: ignore
                        rv_stats.update_trunc(int(idx_keep.shape[0]), int(res.shape[0]))
                        for ii in idx_keep:
                            e = b[int(ii)]
                            key = (e.mr, e.mi)
                            if key not in merged:
                                merged[key] = (e, 1)
                            else:
                                merged[key] = (merged[key][0], merged[key][1] + 1)
                    else:
                        # no robust: no top-m truncation
                        for e in b:
                            key = (e.mr, e.mi)
                            if key not in merged:
                                merged[key] = (e, 1)
                            else:
                                merged[key] = (merged[key][0], merged[key][1] + 1)

                if model.merge_mode == "union":
                    merged = {k: (v[0], 1) for k, v in merged.items()}

            else:
                # baseline: single bucket (buckets[0])
                b = buckets[0] if buckets else []
                if enable_robust and b:
                    gm_list = [np.array(e.g, dtype=np.float32) for e in b if e.g is not None]
                    gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)
                    if gm_arr.shape[0] > 0:
                        res = np.linalg.norm((gm_arr - gs[None, :]) / np.array(
                            [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)], dtype=np.float32
                        ), axis=1)
                        idx_keep = voter.select_top_m(res)  # type: ignore
                        rv_stats.update_trunc(int(idx_keep.shape[0]), int(res.shape[0]))
                        for ii in idx_keep:
                            e = b[int(ii)]
                            merged[(e.mr, e.mi)] = (e, 1)
                    else:
                        pass
                else:
                    for e in b:
                        merged[(e.mr, e.mi)] = (e, 1)

            # vote
            for (mr, mi), (e, cnt) in merged.items():
                alpha = model.alpha_m[mr][mi] - alpha_s
                alpha = wrap_to_pi(alpha)
                bin_j = int(math.floor((alpha + math.pi) / model.angle_step))
                bin_j = max(0, min(aux_size - 1, bin_j))

                if not enable_robust:
                    accumulator[mr, bin_j] += int(cnt) if model.merge_mode == "count" else 1
                    if enable_kde:
                        # store samples with weight=1
                        key = (mr, bin_j)
                        samples.setdefault(key, []).append((alpha, 1.0))
                    continue

                # robust: need residual
                if e.g is None:
                    continue
                gm = np.array(e.g, dtype=np.float32)
                r = voter.residual(gs, gm, model.model_diameter)  # type: ignore
                w = voter.compute_weight(r)  # type: ignore
                if model.merge_mode == "count":
                    w *= float(cnt)
                voter.vote(accumulator, mr, bin_j, w, rv_stats)  # type: ignore

                if enable_kde:
                    key = (mr, bin_j)
                    samples.setdefault(key, []).append((alpha, float(w)))

        # For each reference mr, choose peak bin and generate pose
        row_max = float(np.max(accumulator)) if accumulator.size > 0 else 0.0
        if logger:
            # sr peak distribution summary
            logger.info(f"[SR {sr}] accumulator_global_max={row_max:.3f} candidate_inflation_mean_sofar={float(np.mean(candidate_inflations)):.3f}")

        for mr in range(len(model.pts)):
            row = accumulator[mr]
            votes = float(np.max(row))
            if votes <= 0:
                continue
            bj = int(np.argmax(row))
            theta0 = (float(bj) + 0.5) * model.angle_step - math.pi

            theta = theta0
            if enable_kde and refiner is not None:
                # find top-k bins for this mr
                k = min(refiner.top_k, aux_size)
                if k > 1:
                    top_idx = np.argpartition(-row, k - 1)[:k]
                    # gather samples
                    ts = []
                    ws = []
                    for b in top_idx:
                        key = (mr, int(b))
                        if key in samples:
                            for a, w in samples[key]:
                                ts.append(a)
                                ws.append(w)
                    if len(ts) >= 3:
                        theta, trace = refiner.refine(np.array(ts, dtype=np.float32),
                                                      np.array(ws, dtype=np.float32),
                                                      theta_init=theta0)
                        kde_calls += 1
                        if logger:
                            logger.info(f"[KDERefine] mr={mr} theta0={theta0:.4f} theta={theta:.4f} iters={trace.iters}")

            R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), float(theta))
            T_x = make_affine(R_x, np.zeros(3, dtype=float))

            # T_mg as in baseline
            T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
            T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
            voted_poses.append(PoseWithVotes(T, votes))

        accumulator.fill(0)
        samples.clear()

    best = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
    T_final = best[0].T if best else np.eye(4, dtype=float)

    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "robust_vote": rv_stats.summary() if enable_robust else {},
        "kde_refine_calls": kde_calls
    }
    return T_final, debug


# 对外暴露的完整注册入口，负责读取点云、构建或加载模型、执行注册并整理输出。
def run_registration(
    model_path: str,
    scene_path: str,
    cfg: dict,
    logger: Optional[logging.Logger] = None,
    # ✅ 最小新增：给缓存路径就不再 build_ppf_model
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

    # 读取基础参数并换算角度/距离步长。
    sampling_leaf = float(cfg.get("sampling_leaf", 5.0))
    normal_k = int(cfg.get("normal_k", 5))
    angle_step_deg = float(cfg.get("angle_step_deg", 12.0))
    angle_step = math.radians(angle_step_deg)
    distance_step = float(cfg.get("distance_step_ratio", 0.6)) * sampling_leaf

    t0 = time.perf_counter()

    # read
    # 读取模型点云并检查是否为空。
    cloud_model = o3d.io.read_point_cloud(model_path)
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")
    # 读取场景点云并检查是否为空。
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Empty scene: {scene_path}")

    # ✅ 最小改动：model preprocess + build 改为 “load cache or build”
    # 若提供缓存路径则优先加载缓存，否则在线构建模型。
    model_down = None
    if model_cache_path:
        with Timer("model_load_cache", logger=logger) as tm:
            ppf_model = load_ppf_model_cache(model_cache_path, cfg=cfg, strict=strict_cache)
    else:
        with Timer("model_build", logger=logger) as tm:
            model_down = subsample_and_calculate_normals_model(cloud_model, voxel_size=sampling_leaf, k=normal_k)
            ppf_model = build_ppf_model(model_down, angle_step, distance_step, cfg, logger=logger)

    # scene preprocess + register
    with Timer("registration", logger=logger) as tr:
        scene_down = subsample_and_calculate_normals_scene(cloud_scene, voxel_size=sampling_leaf, k=normal_k)
        T_pred, debug = ppf_register(ppf_model, scene_down, cfg, logger=logger)

    # optional ICP refine on downsampled clouds
    # 如果开启了 ICP 精化，则在注册结果基础上继续优化。
    icp_cfg = cfg.get("icp_refine", {})
    if bool(icp_cfg.get("enable", False)):
        if logger:
            logger.info("[ICP] enabled, refining...")
        # ✅ 缓存模式下也要有 model_down 给 ICP 用
        if model_down is None:
            model_down = subsample_and_calculate_normals_model(cloud_model, voxel_size=sampling_leaf, k=normal_k)
        T_ref = refine_icp_point_to_point(
            source=model_down,
            target=scene_down,
            init_T=T_pred,
            distance_threshold=float(icp_cfg.get("distance_threshold", 5.0)),
            max_iter=int(icp_cfg.get("max_iter", 30))
        )
        T_pred = T_ref

    # output model cloud transformed (downsampled points as in baseline core)
    # 构造输出模型点云，将模型点按预测位姿变换到场景中。
    out_model = o3d.geometry.PointCloud()
    P = (T_pred[:3, :3] @ ppf_model.pts.T).T + T_pred[:3, 3]
    out_model.points = o3d.utility.Vector3dVector(P)

    # 计算整体总耗时。
    total_time = time.perf_counter() - t0

    # 整理并返回统计信息。
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
