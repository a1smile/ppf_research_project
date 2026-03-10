import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .utils import so3_distance, make_affine


@dataclass
class PoseWithVotes:
    T: np.ndarray
    votes: float


def poses_within_error_bounds(T1: np.ndarray, T2: np.ndarray,
                              pos_thresh: float, rot_thresh: float) -> Tuple[bool, float, float]:
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    pos_diff = float(np.linalg.norm(t1 - t2))
    rot_diff = so3_distance(R1, R2)
    return (pos_diff <= pos_thresh and rot_diff <= rot_thresh, pos_diff, rot_diff)


def cluster_poses(poses: List[PoseWithVotes], pos_thresh: float, rot_thresh: float) -> List[PoseWithVotes]:
    poses = sorted(poses, key=lambda pv: pv.votes, reverse=True)
    clusters: List[List[PoseWithVotes]] = []
    cluster_votes: List[float] = []

    for pv in poses:
        found_idx = -1
        best_pos = float("inf")
        best_rot = float("inf")
        for ci, cl in enumerate(clusters):
            ok, pos_d, rot_d = poses_within_error_bounds(pv.T, cl[0].T, pos_thresh, rot_thresh)
            if ok and (pos_d < best_pos and rot_d < best_rot):
                found_idx = ci
                best_pos = pos_d
                best_rot = rot_d

        if found_idx >= 0:
            clusters[found_idx].append(pv)
            cluster_votes[found_idx] += pv.votes
        else:
            clusters.append([pv])
            cluster_votes.append(pv.votes)

    if not clusters:
        return []

    best_votes = max(cluster_votes)
    kept: List[PoseWithVotes] = []

    for cl, v in zip(clusters, cluster_votes):
        if v < 0.1 * best_votes:
            continue

        translations = [c.T[:3, 3] for c in cl]
        t_avg = np.mean(np.stack(translations, axis=0), axis=0)

        # quaternion average (baseline-style)
        quats = []
        for c in cl:
            R = c.T[:3, :3]
            qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
            qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-12)
            qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-12)
            qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-12)
            quats.append(np.array([qw, qx, qy, qz], dtype=float))
        Q = np.mean(np.stack(quats, axis=0), axis=0)
        Q /= (np.linalg.norm(Q) + 1e-12)
        qw, qx, qy, qz = Q

        R_avg = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
        ], dtype=float)

        kept.append(PoseWithVotes(make_affine(R_avg, t_avg), float(v)))

    kept = sorted(kept, key=lambda pv: pv.votes, reverse=True)
    return kept
