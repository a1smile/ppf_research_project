# 导入 math 模块，用于数学计算。
import math
# 导入 dataclass 装饰器，用于定义简单数据结构。
from dataclasses import dataclass
# 导入类型标注工具。
from typing import List, Tuple

# 导入 numpy 并命名为 np，用于数值计算。
import numpy as np

# 从工具模块导入旋转距离和仿射矩阵构造函数。
from .utils import so3_distance, make_affine


# 定义带投票分数的位姿数据结构。
@dataclass
class PoseWithVotes:
    # 4x4 位姿变换矩阵。
    T: np.ndarray
    # 该位姿对应的投票分数。
    votes: float


# 判断两个位姿是否在给定位置和旋转误差阈值内。
def poses_within_error_bounds(T1: np.ndarray, T2: np.ndarray,
                              pos_thresh: float, rot_thresh: float) -> Tuple[bool, float, float]:
    # 提取第一个位姿的平移向量。
    t1 = T1[:3, 3]
    # 提取第二个位姿的平移向量。
    t2 = T2[:3, 3]
    # 提取第一个位姿的旋转矩阵。
    R1 = T1[:3, :3]
    # 提取第二个位姿的旋转矩阵。
    R2 = T2[:3, :3]
    # 计算平移差异。
    pos_diff = float(np.linalg.norm(t1 - t2))
    # 计算旋转差异。
    rot_diff = so3_distance(R1, R2)
    # 返回是否满足阈值，以及具体的位置差和旋转差。
    return (pos_diff <= pos_thresh and rot_diff <= rot_thresh, pos_diff, rot_diff)


# 对候选位姿进行聚类，并保留高投票结果。
def cluster_poses(poses: List[PoseWithVotes], pos_thresh: float, rot_thresh: float) -> List[PoseWithVotes]:
    # 按投票数从高到低排序。
    poses = sorted(poses, key=lambda pv: pv.votes, reverse=True)
    # 保存聚类结果，每个聚类内部是多个 PoseWithVotes。
    clusters: List[List[PoseWithVotes]] = []
    # 保存每个聚类的总投票分数。
    cluster_votes: List[float] = []

    # 遍历所有候选位姿，尝试分配到已有聚类或新建聚类。
    for pv in poses:
        # 记录最匹配的聚类索引，默认未找到。
        found_idx = -1
        # 保存当前最优的位置差。
        best_pos = float("inf")
        # 保存当前最优的旋转差。
        best_rot = float("inf")
        # 遍历已有聚类，并与每个聚类的代表位姿比较。
        for ci, cl in enumerate(clusters):
            ok, pos_d, rot_d = poses_within_error_bounds(pv.T, cl[0].T, pos_thresh, rot_thresh)
            # 如果满足阈值且比当前记录更优，则更新所属聚类。
            if ok and (pos_d < best_pos and rot_d < best_rot):
                found_idx = ci
                best_pos = pos_d
                best_rot = rot_d

        # 如果找到可归属的聚类，则加入并累积投票。
        if found_idx >= 0:
            clusters[found_idx].append(pv)
            cluster_votes[found_idx] += pv.votes
        else:
            # 否则新建一个聚类。
            clusters.append([pv])
            cluster_votes.append(pv.votes)

    # 若没有任何聚类结果，则直接返回空列表。
    if not clusters:
        return []

    # 找到聚类中的最大总投票值。
    best_votes = max(cluster_votes)
    # 用于保存最终保留的聚类代表位姿。
    kept: List[PoseWithVotes] = []

    # 遍历各聚类，根据聚类总票数筛选并计算平均位姿。
    for cl, v in zip(clusters, cluster_votes):
        # 丢弃总票数远低于最佳聚类的结果。
        if v < 0.1 * best_votes:
            continue

        # 提取该聚类中所有位姿的平移向量。
        translations = [c.T[:3, 3] for c in cl]
        # 计算平均平移。
        t_avg = np.mean(np.stack(translations, axis=0), axis=0)

        # 使用四元数平均法对旋转进行聚合，保持与基线实现风格一致。
        quats = []
        for c in cl:
            R = c.T[:3, :3]
            qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
            qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-12)
            qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-12)
            qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-12)
            quats.append(np.array([qw, qx, qy, qz], dtype=float))
        # 对所有四元数取平均。
        Q = np.mean(np.stack(quats, axis=0), axis=0)
        # 将平均后的四元数归一化。
        Q /= (np.linalg.norm(Q) + 1e-12)
        qw, qx, qy, qz = Q

        # 将平均四元数转换回旋转矩阵。
        R_avg = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
        ], dtype=float)

        # 保存聚类后的代表位姿及该聚类总票数。
        kept.append(PoseWithVotes(make_affine(R_avg, t_avg), float(v)))

    # 按票数降序排序最终结果。
    kept = sorted(kept, key=lambda pv: pv.votes, reverse=True)
    # 返回保留的聚类代表位姿列表。
    return kept
