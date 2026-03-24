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
        dr = rotation_angle_rad(R, R_center)
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
        hypos.append(PoseHypothesis(T=T, score=score))
    return hypos


def hypotheses_from_matrices(
    pose_mats: List[np.ndarray],
    scores: Optional[List[float]] = None,
) -> List[PoseHypothesis]:
    if scores is None:
        scores = [1.0] * len(pose_mats)

    assert len(pose_mats) == len(scores)

    hypos: List[PoseHypothesis] = []
    for T, s in zip(pose_mats, scores):
        hypos.append(PoseHypothesis(T=np.asarray(T, dtype=np.float64), score=float(s)))
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
        }

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
            dr = rotation_angle_rad(R, R_rep)
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
    }
    return best_cluster, clusters, debug
