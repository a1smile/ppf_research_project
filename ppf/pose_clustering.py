# 启用延迟类型注解，避免类型在定义前引用时报错。
from __future__ import annotations

# 导入 dataclass，用来简洁定义数据类。
from dataclasses import dataclass
# 导入类型提示工具，便于代码可读性和 IDE 检查。
from typing import List, Optional, Tuple, Dict, Any
# 导入 numpy，做矩阵和数值计算。
import numpy as np


# 定义单个位姿假设的数据结构。
@dataclass
class PoseHypothesis:
    # 4x4 位姿矩阵。
    T: np.ndarray
    # 该位姿的分数，通常对应 vote 数或 confidence。
    score: float = 1.0
    # 可选的额外信息，默认没有。
    meta: Optional[dict] = None


# 定义单个位姿簇的数据结构。
@dataclass
class PoseCluster:
    # 该簇内所有成员在 hypotheses 列表中的索引。
    members: List[int]
    # 该簇所有成员分数之和。
    score_sum: float
    # 代表位姿在 hypotheses 列表中的索引。
    representative_idx: int
    # 代表位姿的 4x4 矩阵。
    T_rep: np.ndarray
    # 该簇平移中心的均值。
    t_mean: np.ndarray
    # 该簇成员数量。
    size: int
    # 该簇中单个成员的最大分数。
    max_score: float


# 计算两个旋转矩阵之间的相对旋转角，单位为弧度。
def rotation_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    # 计算相对旋转矩阵。
    R = R1.T @ R2
    # 计算迹。
    trace_val = np.trace(R)
    # 根据旋转矩阵迹求 cos(theta)。
    cos_theta = (trace_val - 1.0) / 2.0
    # 数值稳定性裁剪，避免 arccos 输入越界。
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算旋转角。
    theta = np.arccos(cos_theta)
    # 返回 python float。
    return float(theta)


# 计算两个平移向量之间的欧氏距离。
def translation_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    # 返回 L2 范数。
    return float(np.linalg.norm(t1 - t2))


# 对一组旋转矩阵做简单平均，返回平均旋转。
def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    # 如果只有一个旋转，直接返回其拷贝。
    if len(rotations) == 1:
        return rotations[0].copy()

    # 初始化 3x3 累加矩阵。
    M = np.zeros((3, 3), dtype=np.float64)

    # 遍历每个旋转矩阵。
    for R in rotations:
        # 直接累加矩阵。
        M += R

    # 对累加矩阵做 SVD 分解。
    U, _, Vt = np.linalg.svd(M)

    # 用极分解方式恢复到最近的正交旋转矩阵。
    R_avg = U @ Vt

    # 若行列式为负，说明出现了反射，需要修正。
    if np.linalg.det(R_avg) < 0:
        # 翻转最后一列。
        U[:, -1] *= -1
        # 重新计算平均旋转。
        R_avg = U @ Vt

    # 返回平均旋转。
    return R_avg


# 根据给定旋转和平移组装 4x4 位姿矩阵。
def build_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # 初始化 4x4 单位矩阵。
    T = np.eye(4, dtype=np.float64)
    # 填入旋转部分。
    T[:3, :3] = R
    # 填入平移部分。
    T[:3, 3] = t
    # 返回完整位姿。
    return T


# 在一个簇内部，选择最接近簇中心的成员作为代表位姿。
def choose_representative_member(
    hypos: List[PoseHypothesis],
    member_indices: List[int],
    t_center: np.ndarray,
    R_center: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
) -> int:
    # 初始默认第一个成员为最佳。
    best_idx = member_indices[0]
    # 初始最小代价设为正无穷。
    best_cost = float("inf")

    # 遍历当前簇内所有成员。
    for idx in member_indices:
        # 取出该成员的位姿矩阵。
        T = hypos[idx].T
        # 取出平移部分。
        t = T[:3, 3]
        # 取出旋转部分。
        R = T[:3, :3]

        # 计算当前成员到平移中心的距离。
        dt = translation_distance(t, t_center)
        # 计算当前成员到旋转中心的角距离。
        dr = rotation_angle_rad(R, R_center)

        # 构造总代价，平移和旋转加权求和。
        cost = pos_weight * dt + rot_weight * dr

        # 如果代价更小，则更新最优成员。
        if cost < best_cost:
            # 更新最优代价。
            best_cost = cost
            # 更新最优索引。
            best_idx = idx

    # 返回代表成员索引。
    return best_idx


# 将 registration.py 里的 voted_poses 转成 pose clustering 使用的 PoseHypothesis 列表。
def hypotheses_from_posewithvotes(voted_poses: List[Any]) -> List[PoseHypothesis]:
    # 初始化输出列表。
    hypos: List[PoseHypothesis] = []

    # 遍历所有带投票数的位姿对象。
    for vp in voted_poses:
        # 尝试读取 votes 属性，若不存在则默认 1.0。
        score = float(getattr(vp, "votes", 1.0))
        # 读取 T 属性并转成 numpy 数组。
        T = np.asarray(getattr(vp, "T"), dtype=np.float64)
        # 构造 PoseHypothesis 并加入列表。
        hypos.append(PoseHypothesis(T=T, score=score))

    # 返回转换后的假设列表。
    return hypos


# 将纯位姿矩阵列表和分数列表转成 PoseHypothesis 列表。
def hypotheses_from_matrices(
    pose_mats: List[np.ndarray],
    scores: Optional[List[float]] = None,
) -> List[PoseHypothesis]:
    # 如果未提供 scores，则全部设为 1.0。
    if scores is None:
        scores = [1.0] * len(pose_mats)

    # 断言长度一致。
    assert len(pose_mats) == len(scores)

    # 初始化输出列表。
    hypos: List[PoseHypothesis] = []

    # 同时遍历位姿和分数。
    for T, s in zip(pose_mats, scores):
        # 将每个元素封装成 PoseHypothesis。
        hypos.append(PoseHypothesis(T=np.asarray(T, dtype=np.float64), score=float(s)))

    # 返回结果。
    return hypos


# 对一组位姿假设进行聚类。
def cluster_pose_hypotheses(
    hypotheses: List[PoseHypothesis],
    pos_thresh: float = 20.0,
    rot_thresh_rad: float = 0.35,
    min_cluster_size: int = 1,
    merge_by_score: bool = True,
) -> Tuple[Optional[PoseCluster], List[PoseCluster], Dict[str, Any]]:
    # 如果输入为空，则直接返回空结果和调试信息。
    if len(hypotheses) == 0:
        # 返回空最佳簇、空簇列表和统计信息。
        return None, [], {
            "num_hypotheses": 0,
            "num_clusters": 0,
            "best_cluster_size": 0,
            "best_cluster_score_sum": 0.0,
        }

    # 构造初始索引列表。
    indices = list(range(len(hypotheses)))

    # 如果需要按分数优先聚类，则先按分数从高到低排序。
    if merge_by_score:
        indices.sort(key=lambda i: hypotheses[i].score, reverse=True)

    # 原始簇列表，每个簇只存成员索引。
    clusters_raw: List[List[int]] = []

    # 逐个处理候选位姿。
    for idx in indices:
        # 取出当前候选位姿。
        T = hypotheses[idx].T
        # 取平移。
        t = T[:3, 3]
        # 取旋转。
        R = T[:3, :3]

        # 标记当前候选是否已经被分到某个簇。
        assigned = False

        # 遍历已有原始簇。
        for members in clusters_raw:
            # 用该簇第一个成员作为临时代表。
            rep_idx = members[0]
            # 取该代表的位姿。
            T_rep = hypotheses[rep_idx].T
            # 取代表平移。
            t_rep = T_rep[:3, 3]
            # 取代表旋转。
            R_rep = T_rep[:3, :3]

            # 计算当前位姿与该簇代表的平移距离。
            dt = translation_distance(t, t_rep)
            # 计算当前位姿与该簇代表的旋转角距离。
            dr = rotation_angle_rad(R, R_rep)

            # 如果平移和旋转都在阈值内，则归入该簇。
            if dt <= pos_thresh and dr <= rot_thresh_rad:
                # 加入该簇。
                members.append(idx)
                # 标记已分配。
                assigned = True
                # 跳出簇遍历。
                break

        # 如果没有分到任何现有簇，则新建一个簇。
        if not assigned:
            # 新簇只含当前索引。
            clusters_raw.append([idx])

    # 初始化最终簇列表。
    clusters: List[PoseCluster] = []

    # 遍历每个原始簇，生成更完整的簇对象。
    for members in clusters_raw:
        # 如果簇大小小于最小簇大小，则丢弃。
        if len(members) < min_cluster_size:
            continue

        # 收集当前簇内所有旋转矩阵。
        member_rotations = [hypotheses[i].T[:3, :3] for i in members]
        # 收集当前簇内所有平移向量，并堆叠成数组。
        member_translations = np.stack([hypotheses[i].T[:3, 3] for i in members], axis=0)
        # 收集当前簇内所有分数。
        member_scores = [float(hypotheses[i].score) for i in members]

        # 计算平移均值作为簇中心。
        t_mean = np.mean(member_translations, axis=0)
        # 计算平均旋转作为簇中心。
        R_mean = average_rotations(member_rotations)

        # 选择离簇中心最近的成员作为代表位姿。
        rep_idx = choose_representative_member(
            hypos=hypotheses,
            member_indices=members,
            t_center=t_mean,
            R_center=R_mean,
            pos_weight=1.0,
            rot_weight=0.2,
        )

        # 构造 PoseCluster 对象。
        cluster = PoseCluster(
            members=members,
            score_sum=float(np.sum(member_scores)),
            representative_idx=rep_idx,
            T_rep=hypotheses[rep_idx].T.copy(),
            t_mean=t_mean,
            size=len(members),
            max_score=float(np.max(member_scores)),
        )

        # 加入最终簇列表。
        clusters.append(cluster)

    # 如果没有任何有效簇，则返回空结果。
    if len(clusters) == 0:
        # 返回空最佳簇、空簇列表和调试信息。
        return None, [], {
            "num_hypotheses": len(hypotheses),
            "num_clusters": 0,
            "best_cluster_size": 0,
            "best_cluster_score_sum": 0.0,
        }

    # 按“总分数、簇大小、最大分数”从大到小排序。
    clusters.sort(key=lambda c: (c.score_sum, c.size, c.max_score), reverse=True)

    # 排序后第一个簇视为最优簇。
    best_cluster = clusters[0]

    # 构造调试信息字典。
    debug = {
        "num_hypotheses": len(hypotheses),
        "num_clusters": len(clusters),
        "best_cluster_size": best_cluster.size,
        "best_cluster_score_sum": best_cluster.score_sum,
        "largest_cluster_size": max(c.size for c in clusters),
        "cluster_sizes_top5": [c.size for c in clusters[:5]],
        "cluster_scores_top5": [c.score_sum for c in clusters[:5]],
    }

    # 返回最优簇、全部簇和调试信息。
    return best_cluster, clusters, debug