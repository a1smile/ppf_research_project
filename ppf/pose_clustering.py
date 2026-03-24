from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PoseHypothesis:
    T: np.ndarray
    score: float = 1.0
    meta: Optional[dict] = None


@dataclass
class PoseCluster:
    members: List[int]
    score_sum: float
    representative_idx: int
    T_rep: np.ndarray
    t_mean: np.ndarray
    size: int
    max_score: float
    mean_score: float
    mode_score: float = 0.0


def rotation_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    trace_val = np.trace(R)
    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def translation_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    return float(np.linalg.norm(t1 - t2))


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0].copy()

    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R

    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def _axis_name_to_vec(axis_name: Any) -> Optional[np.ndarray]:
    if axis_name is None:
        return None

    if isinstance(axis_name, str):
        a = axis_name.strip().lower()
        if a == "x":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if a == "y":
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if a == "z":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return None

    if isinstance(axis_name, (list, tuple, np.ndarray)) and len(axis_name) == 3:
        try:
            v = np.asarray(axis_name, dtype=np.float64)
        except Exception:
            return None
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        return v / n

    return None


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / axis_norm

    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c

    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _symmetry_rotations_from_meta(meta: Optional[dict]) -> List[np.ndarray]:
    """
    从 hypothesis.meta 中解析对称群。

    支持两种形式：
    1) meta["symmetry_rotations"] = [R0, R1, ...]，每个 R 为 3x3
    2) meta["symmetry_axis"] + meta["symmetry_order"]
       例如：
         {"symmetry_axis": "z", "symmetry_order": 2}
       表示绕 z 轴二重旋转对称（包含 I 和 180°）

    若没有提供任何对称信息，则返回 [I]，行为与原实现完全一致。
    """
    identity = np.eye(3, dtype=np.float64)
    if meta is None:
        return [identity]

    rots = meta.get("symmetry_rotations", None)
    if rots is not None:
        out: List[np.ndarray] = [identity]
        for R in rots:
            R = np.asarray(R, dtype=np.float64)
            if R.shape == (3, 3):
                out.append(R)
        return out

    axis_name = meta.get("symmetry_axis", None)
    order = meta.get("symmetry_order", None)
    if axis_name is not None and order is not None:
        try:
            order = int(order)
        except Exception:
            order = 1

        axis_vec = _axis_name_to_vec(axis_name)
        if axis_vec is not None and order >= 2:
            out = []
            for k in range(order):
                angle = 2.0 * np.pi * float(k) / float(order)
                out.append(_rotation_matrix_from_axis_angle(axis_vec, angle))
            return out

    return [identity]


def symmetry_aware_rotation_angle_rad(
    R1: np.ndarray,
    R2: np.ndarray,
    meta1: Optional[dict] = None,
    meta2: Optional[dict] = None,
) -> float:
    """
    对称感知的旋转距离。

    若 hypothesis.meta 中没有提供对称群，则退化为普通 rotation_angle_rad。
    若提供了对象坐标系下的对称旋转 S，则使用：
        d(R1, R2) = min_S angle(R1, R2 @ S)
    这样可以把“只差一个对象内在对称变换”的两个 pose 视为同一模式。
    """
    syms1 = _symmetry_rotations_from_meta(meta1)
    syms2 = _symmetry_rotations_from_meta(meta2)

    # 优先使用非平凡的对称群；若两边都没有，则和原逻辑完全一致。
    if len(syms1) > 1:
        syms = syms1
    elif len(syms2) > 1:
        syms = syms2
    else:
        return rotation_angle_rad(R1, R2)

    best = float("inf")
    for S in syms:
        dr = rotation_angle_rad(R1, R2 @ S)
        if dr < best:
            best = dr
    return best


def choose_representative_member(
    hypos: List[PoseHypothesis],
    member_indices: List[int],
    t_center: np.ndarray,
    R_center: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 1.0,
    score_weight: float = 0.1,
) -> int:
    best_idx = member_indices[0]
    best_cost = float("inf")

    max_member_score = max(float(hypos[i].score) for i in member_indices)
    max_member_score = max(max_member_score, 1e-12)

    for idx in member_indices:
        T = hypos[idx].T
        t = T[:3, 3]
        R = T[:3, :3]

        dt = translation_distance(t, t_center)
        dr = symmetry_aware_rotation_angle_rad(
            R,
            R_center,
            meta1=hypos[idx].meta,
            meta2=hypos[idx].meta,
        )
        score_term = float(hypos[idx].score) / max_member_score
        cost = pos_weight * dt + rot_weight * dr - score_weight * score_term

        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    return best_idx


def hypotheses_from_posewithvotes(voted_poses: List[Any]) -> List[PoseHypothesis]:
    hypos: List[PoseHypothesis] = []
    for vp in voted_poses:
        score = float(getattr(vp, "votes", 1.0))
        T = np.asarray(getattr(vp, "T"), dtype=np.float64)
        meta = getattr(vp, "meta", None)
        hypos.append(PoseHypothesis(T=T, score=score, meta=meta))
    return hypos


def hypotheses_from_matrices(
    pose_mats: List[np.ndarray],
    scores: Optional[List[float]] = None,
    metas: Optional[List[Optional[dict]]] = None,
) -> List[PoseHypothesis]:
    if scores is None:
        scores = [1.0] * len(pose_mats)
    if metas is None:
        metas = [None] * len(pose_mats)

    assert len(pose_mats) == len(scores)
    assert len(pose_mats) == len(metas)

    hypos: List[PoseHypothesis] = []
    for T, s, m in zip(pose_mats, scores, metas):
        hypos.append(PoseHypothesis(T=np.asarray(T, dtype=np.float64), score=float(s), meta=m))
    return hypos


def cluster_pose_hypotheses(
    hypotheses: List[PoseHypothesis],
    pos_thresh: float = 20.0,
    rot_thresh_rad: float = 0.35,
    min_cluster_size: int = 1,
    merge_by_score: bool = True,
    size_weight: float = 0.40,
    mean_weight: float = 0.30,
    max_weight: float = 0.30,
) -> Tuple[Optional[PoseCluster], List[PoseCluster], Dict[str, Any]]:
    if len(hypotheses) == 0:
        return None, [], {
            "num_hypotheses": 0,
            "num_clusters": 0,
            "best_cluster_size": 0,
            "best_cluster_score_sum": 0.0,
            "best_cluster_mode_score": 0.0,
            "symmetry_aware": False,
        }

    symmetry_aware_enabled = any(
        len(_symmetry_rotations_from_meta(h.meta)) > 1 for h in hypotheses
    )

    indices = list(range(len(hypotheses)))
    if merge_by_score:
        indices.sort(key=lambda i: hypotheses[i].score, reverse=True)

    clusters_raw: List[List[int]] = []
    for idx in indices:
        T = hypotheses[idx].T
        t = T[:3, 3]
        R = T[:3, :3]

        assigned = False
        for members in clusters_raw:
            rep_idx = members[0]
            T_rep = hypotheses[rep_idx].T
            t_rep = T_rep[:3, 3]
            R_rep = T_rep[:3, :3]

            dt = translation_distance(t, t_rep)
            dr = symmetry_aware_rotation_angle_rad(
                R,
                R_rep,
                meta1=hypotheses[idx].meta,
                meta2=hypotheses[rep_idx].meta,
            )
            if dt <= pos_thresh and dr <= rot_thresh_rad:
                members.append(idx)
                assigned = True
                break

        if not assigned:
            clusters_raw.append([idx])

    clusters: List[PoseCluster] = []
    for members in clusters_raw:
        if len(members) < min_cluster_size:
            continue

        member_rotations = [hypotheses[i].T[:3, :3] for i in members]
        member_translations = np.stack([hypotheses[i].T[:3, 3] for i in members], axis=0)
        member_scores = [float(hypotheses[i].score) for i in members]

        t_mean = np.mean(member_translations, axis=0)
        R_mean = average_rotations(member_rotations)
        rep_idx = choose_representative_member(
            hypos=hypotheses,
            member_indices=members,
            t_center=t_mean,
            R_center=R_mean,
            pos_weight=1.0,
            rot_weight=0.2,
            score_weight=0.1,
        )

        score_sum = float(np.sum(member_scores))
        mean_score = float(score_sum / max(1, len(members)))
        cluster = PoseCluster(
            members=members,
            score_sum=score_sum,
            representative_idx=rep_idx,
            T_rep=hypotheses[rep_idx].T.copy(),
            t_mean=t_mean,
            size=len(members),
            max_score=float(np.max(member_scores)),
            mean_score=mean_score,
            mode_score=0.0,
        )
        clusters.append(cluster)

    if len(clusters) == 0:
        return None, [], {
            "num_hypotheses": len(hypotheses),
            "num_clusters": 0,
            "best_cluster_size": 0,
            "best_cluster_score_sum": 0.0,
            "best_cluster_mode_score": 0.0,
            "symmetry_aware": symmetry_aware_enabled,
        }

    max_size = max(c.size for c in clusters)
    max_mean = max(c.mean_score for c in clusters)
    max_max = max(c.max_score for c in clusters)
    max_size = max(max_size, 1)
    max_mean = max(max_mean, 1e-12)
    max_max = max(max_max, 1e-12)

    for c in clusters:
        size_term = float(c.size) / float(max_size)
        mean_term = float(c.mean_score) / float(max_mean)
        max_term = float(c.max_score) / float(max_max)
        c.mode_score = (
            float(size_weight) * size_term
            + float(mean_weight) * mean_term
            + float(max_weight) * max_term
        )

    clusters.sort(key=lambda c: (c.mode_score, c.score_sum, c.size, c.max_score), reverse=True)
    best_cluster = clusters[0]

    debug = {
        "num_hypotheses": len(hypotheses),
        "num_clusters": len(clusters),
        "best_cluster_size": best_cluster.size,
        "best_cluster_score_sum": best_cluster.score_sum,
        "best_cluster_mode_score": best_cluster.mode_score,
        "largest_cluster_size": max(c.size for c in clusters),
        "cluster_sizes_top5": [c.size for c in clusters[:5]],
        "cluster_scores_top5": [c.score_sum for c in clusters[:5]],
        "cluster_mean_scores_top5": [c.mean_score for c in clusters[:5]],
        "cluster_mode_scores_top5": [c.mode_score for c in clusters[:5]],
        "mode_score_weights": {
            "size": float(size_weight),
            "mean": float(mean_weight),
            "max": float(max_weight),
        },
        "symmetry_aware": symmetry_aware_enabled,
    }
    return best_cluster, clusters, debug
