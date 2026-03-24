# registration.py
# 中文说明：
# 该模块负责从模型点云与场景点云出发，执行完整 PPF 注册流程，
# 包括：
# 1）模型和场景预处理
# 2）模型构建或缓存加载
# 3）PPF 核心注册
# 4）可选的 ICP 精修
# 5）返回预测位姿、输出点云、调试信息和统计信息

# 导入数学库，用于角度、弧度和三角函数计算。
import math

# 导入时间库，用于统计流程耗时。
import time

# 导入日志库，用于输出运行信息。
import logging

# 导入文件名解析工具。
import os
import re

# 导入数据类装饰器，用于定义结构化结果。
from dataclasses import dataclass

# 导入类型标注工具，便于描述接口。
from typing import Dict, Any, Optional, Tuple, List

# 导入 NumPy，用于数组和矩阵运算。
import numpy as np

# 导入 Open3D，用于点云读写、KDTree 和几何处理。
import open3d as o3d

# 导入通用工具函数，包括随机种子、计时器和刚体变换操作。
from .utils import (
    set_global_seed,
    Timer,
    make_affine,
    invert_affine,
    compose_affine,
    rotation_matrix_from_axis_angle,
    wrap_to_pi,
)

# 导入 PPF 特征计算与角度转换函数。
from .ppf_features import (
    compute_pair_features,
    angle_from_transformed_point,
    to_internal_feature_g,
)

# 导入 PPF 模型构建函数和模型类型。
from .model_builder import build_ppf_model, PPFModel

# 导入固定下采样和自适应下采样接口。
from .preprocess import (
    subsample_and_calculate_normals_model,
    subsample_and_calculate_normals_scene,
    adaptive_subsample_and_calculate_normals_model,
    adaptive_subsample_and_calculate_normals_scene,
)

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
from .pose_clustering import (
    hypotheses_from_posewithvotes,
    hypotheses_from_matrices,
    cluster_pose_hypotheses,
)
from .pose_selection import select_pose_hypotheses


# 计算场景参考点到参考坐标系的刚体变换。
def compute_transform_sg(
    scene_ref_p: np.ndarray,
    scene_ref_n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # 定义目标对齐的 x 轴。
    ex = np.array([1.0, 0.0, 0.0], dtype=float)

    # 定义退化时备用的 y 轴。
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


def _infer_obj_id_from_model_path(model_path: str) -> Optional[int]:
    """
    从模型文件名中解析 obj_id。
    例如：
    - obj_000008.ply -> 8
    - model_12.ply   -> 12
    """
    name = os.path.splitext(os.path.basename(model_path))[0]
    m = re.search(r"(\d+)$", name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _axis_to_vec(axis_val: Any) -> Optional[List[float]]:
    """
    将配置中的 axis 转成三维向量。
    支持：
    - "x" / "y" / "z"
    - [x, y, z]
    """
    if isinstance(axis_val, str):
        a = axis_val.strip().lower()
        if a == "x":
            return [1.0, 0.0, 0.0]
        if a == "y":
            return [0.0, 1.0, 0.0]
        if a == "z":
            return [0.0, 0.0, 1.0]
        return None

    if isinstance(axis_val, (list, tuple)) and len(axis_val) == 3:
        try:
            return [float(axis_val[0]), float(axis_val[1]), float(axis_val[2])]
        except Exception:
            return None

    return None


def _build_symmetry_meta_from_cfg(model_path: str, cfg: dict) -> Optional[dict]:
    """
    从配置中读取当前 obj_id 的对称信息，转换成 pose_clustering.py 可识别的 meta 格式。

    支持的 YAML 格式示例：
    symmetry:
      8:
        axis: z
        order: 2
      10:
        axis: [0, 0, 1]
        order: 2

    或：
    symmetry:
      "8":
        rotations:
          - [[1,0,0],[0,1,0],[0,0,1]]
          - [[-1,0,0],[0,-1,0],[0,0,1]]
    """
    obj_id = _infer_obj_id_from_model_path(model_path)
    if obj_id is None:
        return None

    sym_cfg = cfg.get("symmetry", {})
    if not isinstance(sym_cfg, dict):
        return None

    obj_cfg = sym_cfg.get(obj_id, sym_cfg.get(str(obj_id), None))
    if not isinstance(obj_cfg, dict):
        return None

    meta: Dict[str, Any] = {"obj_id": int(obj_id)}

    axis_vec = _axis_to_vec(obj_cfg.get("axis", None))
    order_val = obj_cfg.get("order", None)
    if axis_vec is not None and order_val is not None:
        try:
            meta["symmetry_axis"] = axis_vec
            meta["symmetry_order"] = int(order_val)
        except Exception:
            pass

    rots_cfg = obj_cfg.get("rotations", None)
    if isinstance(rots_cfg, list):
        rots = []
        for R in rots_cfg:
            try:
                R_np = np.asarray(R, dtype=np.float64)
            except Exception:
                continue
            if R_np.shape == (3, 3):
                rots.append(R_np)
        if len(rots) > 0:
            meta["symmetry_rotations"] = rots

    return meta if len(meta) > 1 else None


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

    # 从运行时配置中读取当前对象的对称信息（若存在）。
    symmetry_meta = cfg.get("_runtime_symmetry_meta", None)

    # 读取姿态聚类配置。
    pose_cluster_cfg = cfg.get("pose_clustering", {})
    enable_pose_clustering = bool(pose_cluster_cfg.get("enable", True))
    pose_cluster_pos_thresh = float(pose_cluster_cfg.get("pos_thresh", 20.0))
    pose_cluster_rot_thresh_deg = float(pose_cluster_cfg.get("rot_thresh_deg", 20.0))
    pose_cluster_rot_thresh_rad = math.radians(pose_cluster_rot_thresh_deg)
    pose_cluster_min_size = int(pose_cluster_cfg.get("min_cluster_size", 1))
    pose_cluster_merge_by_score = bool(pose_cluster_cfg.get("merge_by_score", True))
    pose_cluster_score_cfg = pose_cluster_cfg.get("score_weights", {})
    pose_cluster_size_weight = float(pose_cluster_score_cfg.get("size", 0.40))
    pose_cluster_mean_weight = float(pose_cluster_score_cfg.get("mean", 0.30))
    pose_cluster_max_weight = float(pose_cluster_score_cfg.get("max", 0.30))

    # 读取 pose selection 配置总开关。
    pose_select_cfg = cfg.get("pose_selection", {})
    enable_pose_selection = bool(pose_select_cfg.get("enable", False))

    # 读取 KDE 配置。
    kde_cfg = cfg.get("kde_refine", {})

    # 最小修改：新增只对 votes 最高前 top-k 个 pose 做 KDE refine 的参数。
    # 若配置里没写，则默认给一个很大的值，等价于原先“几乎全部 refine”。
    refine_top_pose_k = int(kde_cfg.get("refine_top_pose_k", 1000000000))

    # 按配置按需构造鲁棒投票器和 KDE 精修器。
    voter = RobustVoter(cfg.get("robust_vote", {}), logger=logger) if enable_robust else None
    refiner = KDEMeanShiftRefiner(kde_cfg, logger=logger) if enable_kde else None

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
    samples = {}

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
            merged = {}

            # 若启用 RSMRQ，则对每个桶分别处理后再合并。
            if enable_rsmrq:
                for b in buckets:
                    if not b:
                        continue

                    if enable_robust:
                        gm_list = []
                        for e in b:
                            if e.g is None:
                                continue
                            gm_list.append(np.array(e.g, dtype=np.float32))

                        if len(gm_list) != len(b):
                            pass

                        gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)

                        if gm_arr.shape[0] == 0:
                            continue

                        res = np.linalg.norm(
                            (gm_arr - gs[None, :]) / np.array(
                                [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)],
                                dtype=np.float32
                            ),
                            axis=1
                        )

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
                    gm_list = [np.array(e.g, dtype=np.float32) for e in b if e.g is not None]
                    gm_arr = np.stack(gm_list, axis=0) if gm_list else np.zeros((0, 4), dtype=np.float32)

                    if gm_arr.shape[0] > 0:
                        res = np.linalg.norm(
                            (gm_arr - gs[None, :]) / np.array(
                                [math.pi, math.pi, math.pi, max(1e-12, model.model_diameter)],
                                dtype=np.float32
                            ),
                            axis=1
                        )

                        idx_keep = voter.select_top_m(res)  # type: ignore
                        rv_stats.update_trunc(int(idx_keep.shape[0]), int(res.shape[0]))

                        for ii in idx_keep:
                            e = b[int(ii)]
                            merged[(e.mr, e.mi)] = (e, 1)
                else:
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
                    accumulator[mr, bin_j] += int(cnt) if model.merge_mode == "count" else 1

                    if enable_kde:
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

                # 若启用 KDE，则缓存样本角度和权重。
                if enable_kde:
                    key = (mr, bin_j)
                    samples.setdefault(key, []).append((alpha, float(w)))

        # 每个场景参考点处理完后，读取当前累加器全局最大值。
        row_max = float(np.max(accumulator)) if accumulator.size > 0 else 0.0

        # 输出当前参考点的中间统计。
        if logger:
            logger.info(
                f"[SR {sr}] accumulator_global_max={row_max:.3f} "
                f"candidate_inflation_mean_sofar={float(np.mean(candidate_inflations)):.3f}"
            )

        # ------------------------------
        # 最小修改开始：先统计所有 mr 的 votes，再只对 top-k 个 mr 做 KDE refine
        # ------------------------------

        # 统计每个 mr 的最大投票值。
        mr_votes = np.zeros(len(model.pts), dtype=np.float64)
        for mr in range(len(model.pts)):
            row = accumulator[mr]
            mr_votes[mr] = float(np.max(row))

        # 找到有效候选 mr（votes > 0）。
        valid_mrs = np.where(mr_votes > 0)[0]

        # 默认不 refine 任何 mr。
        refine_mr_set = set()

        # 若启用 KDE，则只对 votes 最高的前 refine_top_pose_k 个 mr 做 refine。
        if enable_kde and refiner is not None and len(valid_mrs) > 0:
            valid_votes = mr_votes[valid_mrs]
            k_refine = min(refine_top_pose_k, len(valid_mrs))

            if k_refine < len(valid_mrs):
                top_local_idx = np.argpartition(-valid_votes, k_refine - 1)[:k_refine]
                top_mrs = valid_mrs[top_local_idx]
            else:
                top_mrs = valid_mrs

            refine_mr_set = set(int(x) for x in top_mrs)

            if logger:
                logger.info(
                    f"[KDE TopK] sr={sr} valid={len(valid_mrs)} refine={k_refine}"
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

            # 最小修改：只对 top-k 的 mr 做 KDE refine。
            if enable_kde and refiner is not None and (mr in refine_mr_set):
                k = min(refiner.top_k, aux_size)

                if k > 1:
                    top_idx = np.argpartition(-row, k - 1)[:k]

                    ts = []
                    ws = []

                    for b in top_idx:
                        key = (mr, int(b))
                        if key in samples:
                            for a, w in samples[key]:
                                ts.append(a)
                                ws.append(w)

                    if len(ts) >= 3:
                        theta, trace = refiner.refine(
                            np.array(ts, dtype=np.float32),
                            np.array(ws, dtype=np.float32),
                            theta_init=theta0
                        )
                        kde_calls += 1

                        if logger:
                            logger.info(
                                f"[KDERefine] mr={mr} theta0={theta0:.4f} "
                                f"theta={theta:.4f} iters={trace.iters}"
                            )

            # 构造绕 x 轴的旋转矩阵。
            R_x = rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=float),
                float(theta)
            )

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

    # 先做 pose selection，再做姿态模式聚类；若失败则回退到旧逻辑。
    pose_selection_debug = {}
    pose_cluster_debug = {}

    # 初始化最终姿态。
    T_final = None

    # 先在候选姿态上做 multi-cue pose selection + top-k light refine。
    selected_hypotheses = []
    if enable_pose_selection and len(voted_poses) > 0:
        selected_hypotheses, pose_selection_debug = select_pose_hypotheses(
            voted_poses=voted_poses,
            model_pts=model.pts,
            model_normals=model.normals,
            scene_pcd=scene_pcd,
            cfg=cfg,
            logger=logger,
        )

        # 若存在对称信息，则把它透传给 pose selection 产生的 hypotheses。
        if symmetry_meta is not None:
            for h in selected_hypotheses:
                base_meta = h.meta if isinstance(getattr(h, "meta", None), dict) else {}
                merged_meta = dict(base_meta)
                merged_meta.update(symmetry_meta)
                h.meta = merged_meta

        if logger:
            logger.info(
                "[PoseSelection] "
                f"enabled={enable_pose_selection} "
                f"num_input={pose_selection_debug.get('num_input_candidates', 0)} "
                f"num_preselected={pose_selection_debug.get('num_preselected', 0)} "
                f"num_selected={pose_selection_debug.get('num_selected', 0)} "
                f"num_light_refined={pose_selection_debug.get('num_light_refined', 0)} "
                f"best_score={pose_selection_debug.get('best_score', 0.0):.4f} "
                f"best_inlier={pose_selection_debug.get('best_inlier_ratio', 0.0):.4f}"
            )

    # 若 pose selection 成功产生候选，则优先用这些候选做模式聚类或直接选第一名。
    if len(selected_hypotheses) > 0:
        if enable_pose_clustering:
            best_cluster, clusters_pc, pose_cluster_debug = cluster_pose_hypotheses(
                selected_hypotheses,
                pos_thresh=pose_cluster_pos_thresh,
                rot_thresh_rad=pose_cluster_rot_thresh_rad,
                min_cluster_size=pose_cluster_min_size,
                merge_by_score=pose_cluster_merge_by_score,
                size_weight=pose_cluster_size_weight,
                mean_weight=pose_cluster_mean_weight,
                max_weight=pose_cluster_max_weight,
            )

            if logger:
                logger.info(
                    "[PoseClustering] "
                    f"enabled={enable_pose_clustering} "
                    f"num_hypotheses={pose_cluster_debug.get('num_hypotheses', 0)} "
                    f"num_clusters={pose_cluster_debug.get('num_clusters', 0)} "
                    f"best_cluster_size={pose_cluster_debug.get('best_cluster_size', 0)} "
                    f"best_cluster_score_sum={pose_cluster_debug.get('best_cluster_score_sum', 0.0):.3f} "
                    f"best_cluster_mode_score={pose_cluster_debug.get('best_cluster_mode_score', 0.0):.4f}"
                )

            if best_cluster is not None:
                T_final = best_cluster.T_rep
        else:
            T_final = selected_hypotheses[0].T

    # 若新逻辑未得到结果，则回退到原始姿态聚类。
    if T_final is None and enable_pose_clustering and len(voted_poses) > 0:
        # 若存在对称信息，则直接构造带 meta 的 hypotheses；否则保持原逻辑。
        if symmetry_meta is not None:
            hypos = hypotheses_from_matrices(
                pose_mats=[vp.T for vp in voted_poses],
                scores=[float(getattr(vp, "votes", 1.0)) for vp in voted_poses],
                metas=[dict(symmetry_meta) for _ in voted_poses],
            )
        else:
            hypos = hypotheses_from_posewithvotes(voted_poses)

        best_cluster, clusters_pc, pose_cluster_debug = cluster_pose_hypotheses(
            hypos,
            pos_thresh=pose_cluster_pos_thresh,
            rot_thresh_rad=pose_cluster_rot_thresh_rad,
            min_cluster_size=pose_cluster_min_size,
            merge_by_score=pose_cluster_merge_by_score,
            size_weight=pose_cluster_size_weight,
            mean_weight=pose_cluster_mean_weight,
            max_weight=pose_cluster_max_weight,
        )

        if logger:
            logger.info(
                "[PoseClustering][FallbackRawVotes] "
                f"num_hypotheses={pose_cluster_debug.get('num_hypotheses', 0)} "
                f"num_clusters={pose_cluster_debug.get('num_clusters', 0)} "
                f"best_cluster_mode_score={pose_cluster_debug.get('best_cluster_mode_score', 0.0):.4f}"
            )

        if best_cluster is not None:
            T_final = best_cluster.T_rep

    # 若姿态聚类仍未得到结果，则回退到最原始的 cluster_poses。
    if T_final is None:
        best = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
        T_final = best[0].T if best else np.eye(4, dtype=float)

    # 汇总调试信息。
    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "robust_vote": rv_stats.summary() if enable_robust else {},
        "kde_refine_calls": kde_calls,
        "pose_selection": pose_selection_debug,
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
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    """
    Paper-grade entry point:
    - preprocess model/scene
    - build model hash or load cache
    - run registration
    - optional ICP refinement
    """

    # 读取随机种子并设置全局随机状态。
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    # 读取基础参数。
    sampling_leaf = float(cfg.get("sampling_leaf", 5.0))
    normal_k = int(cfg.get("normal_k", 5))
    angle_step_deg = float(cfg.get("angle_step_deg", 12.0))
    angle_step = math.radians(angle_step_deg)

    # 中文说明：
    # distance_step 依然与 sampling_leaf 绑定，
    # 这样可以最大程度保持与原始模型缓存、原始构建逻辑的一致性。
    distance_step = float(cfg.get("distance_step_ratio", 0.6)) * sampling_leaf

    # 读取自适应下采样总开关。
    adaptive_downsample = bool(cfg.get("adaptive_downsample", False))

    # 读取自适应下采样作用范围。
    # 建议默认只用 scene，避免直接影响模型缓存一致性。
    adaptive_apply_to = str(cfg.get("adaptive_downsample_apply_to", "scene")).lower()

    # 读取自适应下采样子配置。
    adaptive_cfg = cfg.get("adaptive_downsample_cfg", {})

    # 记录总流程起始时间。
    t0 = time.perf_counter()

    # 读取模型点云。
    cloud_model = o3d.io.read_point_cloud(model_path)

    # 若模型点云为空，则报错。
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")

    # 读取场景点云。
    cloud_scene = o3d.io.read_point_cloud(scene_path)

    # 若场景点云为空，则报错。
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Empty scene: {scene_path}")

    # 初始化模型下采样点云。
    model_down = None

    # 初始化模型下采样调试信息。
    model_ds_info: Dict[str, Any] = {
        "raw_points": len(cloud_model.points),
        "down_points": None,
        "target_points": None,
        "voxel_used": sampling_leaf,
        "adaptive_enabled": adaptive_downsample and adaptive_apply_to in ("model", "both"),
        "apply_to": adaptive_apply_to,
    }

    # 初始化场景下采样调试信息。
    scene_ds_info: Dict[str, Any] = {
        "raw_points": len(cloud_scene.points),
        "down_points": None,
        "target_points": None,
        "voxel_used": sampling_leaf,
        "adaptive_enabled": adaptive_downsample and adaptive_apply_to in ("scene", "both"),
        "apply_to": adaptive_apply_to,
    }

    # 若提供缓存路径，则优先加载缓存模型。
    if model_cache_path:
        with Timer("model_load_cache", logger=logger) as tm:
            ppf_model = load_ppf_model_cache(model_cache_path, cfg=cfg, strict=strict_cache)
    else:
        # 否则先做模型预处理，再构建 PPF 模型。
        with Timer("model_build", logger=logger) as tm:
            # 若启用自适应且 apply_to 包含 model，则对模型执行自适应预处理。
            if adaptive_downsample and adaptive_apply_to in ("model", "both"):
                model_down, model_ds_info = adaptive_subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    k=normal_k,
                    cfg=adaptive_cfg,
                )
                model_ds_info["adaptive_enabled"] = True
                model_ds_info["apply_to"] = adaptive_apply_to
            else:
                # 否则保留原始固定 voxel 预处理方式。
                model_down = subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    voxel_size=sampling_leaf,
                    k=normal_k,
                )
                model_ds_info = {
                    "raw_points": len(cloud_model.points),
                    "down_points": len(model_down.points),
                    "target_points": None,
                    "voxel_used": sampling_leaf,
                    "adaptive_enabled": False,
                    "apply_to": adaptive_apply_to,
                }

            # 基于模型下采样结果构建 PPF 模型。
            ppf_model = build_ppf_model(
                model_down,
                angle_step,
                distance_step,
                cfg,
                logger=logger
            )

    # 根据模型路径从配置里提取当前对象的对称信息，并透传到运行时配置。
    cfg_runtime = dict(cfg)
    cfg_runtime["_runtime_symmetry_meta"] = _build_symmetry_meta_from_cfg(model_path, cfg)


    # 对场景点云做预处理并执行注册。
    with Timer("registration", logger=logger) as tr:
        # 若启用自适应且 apply_to 包含 scene，则对 scene 使用自适应预处理。
        if adaptive_downsample and adaptive_apply_to in ("scene", "both"):
            scene_down, scene_ds_info = adaptive_subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                k=normal_k,
                cfg=adaptive_cfg,
            )
            scene_ds_info["adaptive_enabled"] = True
            scene_ds_info["apply_to"] = adaptive_apply_to
        else:
            # 否则沿用原始固定 voxel 预处理。
            scene_down = subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                voxel_size=sampling_leaf,
                k=normal_k,
            )
            scene_ds_info = {
                "raw_points": len(cloud_scene.points),
                "down_points": len(scene_down.points),
                "target_points": None,
                "voxel_used": sampling_leaf,
                "adaptive_enabled": False,
                "apply_to": adaptive_apply_to,
            }

        # 记录调试日志，便于你核查每个样本是否真的被自适应控制了。
        if logger:
            logger.info(f"[AdaptiveDS][model] {model_ds_info}")
            logger.info(f"[AdaptiveDS][scene] {scene_ds_info}")

        # 执行核心 PPF 注册。
        T_pred, debug = ppf_register(ppf_model, scene_down, cfg_runtime, logger=logger)

        # 将下采样调试信息附加到 debug 中，方便后续写 json 或日志时分析。
        debug["model_downsample"] = model_ds_info
        debug["scene_downsample"] = scene_ds_info

    # 按需在下采样点云上执行 ICP 精修。
    icp_cfg = cfg.get("icp_refine", {})

    if bool(icp_cfg.get("enable", False)):
        if logger:
            logger.info("[ICP] enabled, refining...")

        # 若走缓存模式且 model_down 为空，则补做一次模型预处理供 ICP 使用。
        if model_down is None:
            if adaptive_downsample and adaptive_apply_to in ("model", "both"):
                model_down, _ = adaptive_subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    k=normal_k,
                    cfg=adaptive_cfg,
                )
            else:
                model_down = subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    voxel_size=sampling_leaf,
                    k=normal_k,
                )

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
