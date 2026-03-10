# -*- coding: utf-8 -*-
"""
Baseline PPF (Point Pair Feature) registration in Python + Open3D
-----------------------------------------------------------------
This is a *strict* baseline rewrite of the PCL-based C++ program, keeping
the same stages, parameters and even some of the intentional quirks so
that future comparisons are apples-to-apples.

Main characteristics preserved from the original:
- Voxel downsample leaf size = 5.0 (same as sampling_leaf).
- Normal estimation with K=5.
- Model-side normal flipping based on centroid "viewpoint", but referencing
  the ORIGINAL cloud's i-th point when deciding orientation (as in the C++).
- PPF feature definition via PFH's computePairFeatures (f1..f4) + alpha_m.
- Hash map discretization with angle_step = 12 degrees and distance_step = 0.6*leaf.
- Scene reference sampling rate = 20; position clustering = 0.005;
  rotation clustering = 30 degrees (in radians).
- Hough-style accumulator over alpha bins; 90% peak selection.
- Timing printouts and a simple "matching_rate" evaluation.
- Visualization of scene and registered model (random color) with Open3D.

NOTE: This is an educational, baseline-grade implementation. It favors clarity
and structural fidelity over performance. O(N^2) loops are intentional here.
"""
import math
import time
import random
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import open3d as o3d


# ----------------------------- Utility Math -----------------------------


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula for 3x3 R, axis must be unit length."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = ax
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                  [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                  [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=float)
    return R


def make_affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack 3x3 R and 3x1 t into a 4x4 transform (Affine)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_affine(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid 4x4 (R|t; 0 0 0 1)."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_affine(R_inv, t_inv)


def compose_affine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compose rigid transforms: A @ B."""
    return A @ B


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 rigid transform to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


def so3_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations (0..pi)."""
    R = R1.T @ R2
    tr = np.trace(R)
    val = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    return math.acos(val)


# ---------------------- PFH-style pair features (PCL-like) ----------------------


def compute_pair_features(p1: np.ndarray, n1: np.ndarray,
                          p2: np.ndarray, n2: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Emulates PCL's pfh_tools::computePairFeatures signature used by PPFRegistration.
    Returns (f1, f2, f3, f4) or None if degenerate.
    f1 in [-pi, pi]; f2, f3 in [-1, 1]; f4 = distance >= 0.
    """
    dp = p2 - p1
    f4 = float(np.linalg.norm(dp))
    if f4 <= 1e-9:
        return None

    d = dp / f4
    n1n = n1 / (np.linalg.norm(n1) + 1e-12)
    n2n = n2 / (np.linalg.norm(n2) + 1e-12)

    # 若 d 与 n1 几乎平行，跳过
    if abs(float(n1n @ d)) > 0.999:
        return None

    u = n1n
    v = np.cross(u, d)
    nv = np.linalg.norm(v)
    if nv <= 1e-12:
        return None
    v /= nv
    w = np.cross(u, v)

    f1 = math.atan2(float(w @ n2n), float(u @ n2n))
    f2 = float(v @ n2n)
    f3 = float(u @ d)
    # f4 already set
    return (f1, f2, f3, f4)


def angle_from_transformed_point(vec_yzx: np.ndarray) -> float:
    """
    Compute angle = atan2(-z, y), then adjust sign as PCL does.
    """
    y = float(vec_yzx[1])
    z = float(vec_yzx[2])
    ang = math.atan2(-z, y)
    if math.sin(ang) * z < 0.0:
        ang *= -1.0
    return ang


# ----------------------- Downsample + Normals (scene/model) -----------------------


def subsample_and_calculate_normals_scene(pcd: o3d.geometry.PointCloud,
                                          voxel_size: float = 5.0,
                                          k: int = 5) -> o3d.geometry.PointCloud:
    cloud_down = pcd.voxel_down_sample(voxel_size)
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    # Keep as-is (no orientation edit)
    print(f"Cloud dimensions before / after subsampling: {len(pcd.points)} / {len(cloud_down.points)}")
    return cloud_down


def subsample_and_calculate_normals_model(pcd: o3d.geometry.PointCloud,
                                          voxel_size: float = 5.0,
                                          k: int = 5) -> o3d.geometry.PointCloud:
    """
    Mirrors the C++: flip normals toward "viewpoint" = centroid of the SUBSAMPLED cloud,
    but uses the ORIGINAL cloud's i-th point in the dot product decision (quirk kept).
    """
    cloud_down = pcd.voxel_down_sample(voxel_size)
    cloud_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    print(f"Cloud dimensions before / after subsampling: {len(pcd.points)} / {len(cloud_down.points)}")

    pts_down = np.asarray(cloud_down.points)
    if pts_down.shape[0] == 0:
        return cloud_down

    centroid = pts_down.mean(axis=0)
    normals = np.asarray(cloud_down.normals).copy()
    orig_pts = np.asarray(pcd.points)

    for i in range(len(normals)):
        # Replicate the original referencing of cloud->points[i]
        if i < len(orig_pts):
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid  # fallback

        n = normals[i]
        n_norm = np.linalg.norm(n)
        if n_norm == 0.0:
            n = orientation_reference
            n_norm = np.linalg.norm(n)
            if n_norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
        else:
            if float(n @ orientation_reference) > 0.0:
                n = -n
        normals[i] = n

    cloud_down.normals = o3d.utility.Vector3dVector(normals)
    return cloud_down


# ----------------------------- PPF Hash Map (baseline) -----------------------------


@dataclass
class PPFModel:
    angle_step: float
    distance_step: float
    alpha_m: List[List[float]]          # alpha_m[i][j]
    hashmap: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]]
    model_diameter: float               # max pair distance
    ref_R: List[np.ndarray]             # cached R_mg for each reference
    pts: np.ndarray
    normals: np.ndarray


def discretize_feature(f1: float, f2: float, f3: float, f4: float,
                       angle_step: float, distance_step: float) -> Tuple[int, int, int, int]:
    # f1 in [-pi, pi]
    k1 = int(math.floor((f1 + math.pi) / angle_step))
    # f2, f3 are cos-like in [-1, 1]: convert to [0, pi] via acos then bin
    k2 = int(math.floor(math.acos(max(-1.0, min(1.0, f2))) / angle_step))
    k3 = int(math.floor(math.acos(max(-1.0, min(1.0, f3))) / angle_step))
    k4 = int(math.floor(f4 / distance_step))
    return (k1, k2, k3, k4)


def build_ppf_model(model_pcd: o3d.geometry.PointCloud,
                    angle_step: float,
                    distance_step: float) -> PPFModel:
    pts = np.asarray(model_pcd.points).astype(np.float64)
    normals = np.asarray(model_pcd.normals).astype(np.float64)
    N = len(pts)
    alpha_m = [[0.0 for _ in range(N)] for __ in range(N)]
    hashmap: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = defaultdict(list)
    model_diameter = 0.0

    # Precompute R_mg for each reference (align normal to +X), as in PCL:
    ref_R: List[np.ndarray] = [None] * N  # type: ignore
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)

    for i in range(N):
        ni = normals[i]
        ni = ni / (np.linalg.norm(ni) + 1e-12)
        axis = np.cross(ni, ex)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            axis = ey.copy()
            axis_norm = 1.0
        axis /= axis_norm
        angle = math.acos(max(-1.0, min(1.0, float(ni @ ex))))
        ref_R[i] = rotation_matrix_from_axis_angle(axis, angle)

    # Enumerate all ordered pairs (i, j), i != j
    for i in range(N):
        pi = pts[i]
        ni = normals[i]
        Ri = ref_R[i]
        for j in range(N):
            if i == j:
                continue
            pj = pts[j]
            nj = normals[j]

            feat = compute_pair_features(pi, ni, pj, nj)
            if feat is None:
                continue
            f1, f2, f3, f4 = feat
            # 只过滤距离太近的点对，防止数值不稳定
            if f4 < distance_step * 0.5:
                continue
            key = discretize_feature(f1, f2, f3, f4, angle_step, distance_step)
            hashmap[key].append((i, j))

            # Compute alpha_m: transform pj with T_mg (R*(x - pi)), then alpha_m = -atan2(-z, y) with sign rule
            pj_mg = Ri @ (pj - pi)
            angle = angle_from_transformed_point(pj_mg)
            alpha_m[i][j] = -angle

            # 记录最大点对距离作为 model_diameter
            if f4 > model_diameter:
                model_diameter = f4

    return PPFModel(angle_step, distance_step, alpha_m, hashmap, model_diameter, ref_R, pts, normals)


# ----------------------------- PPF Registration (baseline) -----------------------------


@dataclass
class PoseWithVotes:
    T: np.ndarray
    votes: int


def compute_transform_sg(scene_ref_p: np.ndarray, scene_ref_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (T_sg, R_sg) where T_sg = Translation(R * (-p)) * R and R rotates n -> +X.
    """
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)
    n = scene_ref_n / (np.linalg.norm(scene_ref_n) + 1e-12)
    axis = np.cross(n, ex)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        axis = ey.copy()
        axis_norm = 1.0
    axis /= axis_norm
    angle = math.acos(max(-1.0, min(1.0, float(n @ ex))))
    R = rotation_matrix_from_axis_angle(axis, angle)
    t = R @ (-scene_ref_p)
    T = make_affine(R, t)
    return T, R


def nearest_neighbor_search(hashmap: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]],
                            f1: float, f2: float, f3: float, f4: float,
                            angle_step: float, distance_step: float) -> List[Tuple[int, int]]:
    key = discretize_feature(f1, f2, f3, f4, angle_step, distance_step)
    return hashmap.get(key, [])


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
    # Sort by votes descending
    poses = sorted(poses, key=lambda pv: pv.votes, reverse=True)
    clusters: List[List[PoseWithVotes]] = []
    cluster_votes: List[int] = []

    for pv in poses:
        found_idx = -1
        best_pos = float('inf')
        best_rot = float('inf')
        for ci, cl in enumerate(clusters):
            ok, pos_d, rot_d = poses_within_error_bounds(pv.T, cl[0].T, pos_thresh, rot_thresh)
            if ok:
                if pos_d < best_pos and rot_d < best_rot:
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

    # Keep clusters with >= 10% of best-votes
    best_votes = max(cluster_votes)
    kept: List[PoseWithVotes] = []
    for cl, v in zip(clusters, cluster_votes):
        if v >= 0.1 * best_votes:
            # Average rotation via quaternion and average translation
            translations = [c.T[:3, 3] for c in cl]
            t_avg = np.mean(np.stack(translations, axis=0), axis=0)

            # quaternion average
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

            # back to rotation
            qw, qx, qy, qz = Q
            R_avg = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
            ], dtype=float)

            T_avg = make_affine(R_avg, t_avg)
            kept.append(PoseWithVotes(T_avg, v))

    # Sort back by votes
    kept = sorted(kept, key=lambda pv: pv.votes, reverse=True)
    return kept


def ppf_register(model: PPFModel,
                 scene_pcd: o3d.geometry.PointCloud,
                 scene_ref_sampling_rate: int = 20,
                 pos_thresh: float = 0.005,
                 rot_thresh_rad: float = math.radians(30.0)) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Core PPF registration loop, mimicking PCL's PPFRegistration::computeTransformation.
    Returns transformed source (downsampled normals) and final 4x4 transform.
    """
    scene_pts = np.asarray(scene_pcd.points).astype(np.float64)
    scene_normals = np.asarray(scene_pcd.normals).astype(np.float64)
    Nscene = len(scene_pts)

    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    accumulator = np.zeros((len(model.pts), aux_size), dtype=np.int32)

    # KD-tree for scene radius search (radius = model_diameter/2)
    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    radius = float(model.model_diameter) / 2.0

    voted_poses: List[PoseWithVotes] = []

    for sr in range(0, Nscene, scene_ref_sampling_rate):
        sr_p = scene_pts[sr]
        sr_n = scene_normals[sr]

        T_sg, R_sg = compute_transform_sg(sr_p, sr_n)

        # radius neighbors
        _, idxs, _ = kdtree.search_radius_vector_3d(scene_pcd.points[sr], radius)
        for si in idxs:
            if si == sr:
                continue
            si_p = scene_pts[si]
            si_n = scene_normals[si]

            feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
            if feat is None:
                continue
            f1, f2, f3, f4 = feat

            matches = nearest_neighbor_search(model.hashmap, f1, f2, f3, f4,
                                              model.angle_step, model.distance_step)

            # Compute alpha_s
            si_trans = R_sg @ (si_p - sr_p)  # equivalent to T_sg * point
            ang = angle_from_transformed_point(si_trans)
            alpha_s = -ang  # as in PCL

            for (mr, mi) in matches:
                alpha = model.alpha_m[mr][mi] - alpha_s
                # wrap to [-pi, pi]
                if alpha < -math.pi:
                    alpha += 2.0 * math.pi
                elif alpha > math.pi:
                    alpha -= 2.0 * math.pi
                bin_j = int(math.floor((alpha + math.pi) / model.angle_step))
                bin_j = max(0, min(aux_size - 1, bin_j))
                accumulator[mr, bin_j] += 1

        # For each reference mr, find the best alpha bin
        for mr in range(len(model.pts)):
            row = accumulator[mr]
            votes = int(np.max(row))
            if votes <= 0:
                continue
            bj = int(np.argmax(row))
            angle = (float(bj) + 0.5) * model.angle_step - math.pi
            R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), angle)
            T_x = make_affine(R_x, np.zeros(3, dtype=float))

            T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
            T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
            voted_poses.append(PoseWithVotes(T, votes))

        # reset accumulator for next sr
        accumulator.fill(0)

    # Cluster poses and choose the best
    best = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
    if not best:
        # fallback: identity
        T_final = np.eye(4, dtype=float)
    else:
        T_final = best[0].T

    # Apply to the ORIGINAL model cloud (not just downsampled) for evaluation/visualization
    out_model = o3d.geometry.PointCloud()
    out_model.points = o3d.utility.Vector3dVector(transform_points(T_final, model.pts))
    return out_model, T_final


# ----------------------------- Matching rate (baseline) -----------------------------


def matching_rate(model_orig: o3d.geometry.PointCloud,
                  transformed: o3d.geometry.PointCloud,
                  radius: float = 5.0) -> float:
    """
    Fraction of original model points that have at least one neighbor within radius
    in the transformed cloud.
    """
    kd = o3d.geometry.KDTreeFlann(transformed)
    pts = np.asarray(model_orig.points)
    hit = 0
    for i in range(len(pts)):
        k, _, _ = kd.search_radius_vector_3d(pts[i], radius)
        if k > 0:
            hit += 1
    return 100.0 * float(hit) / max(1, len(pts))


# --------------------------------- Top-level API ---------------------------------


def run_baseline_registration(model_path: str,
                              scene_path: str,
                              sampling_leaf: float = 5.0,
                              angle_step_deg: float = 12.0,
                              scene_ref_sampling_rate: int = 20,
                              pos_thresh: float = 0.005,
                              rot_thresh_deg: float = 30.0,
                              registration_runs: int = 1
                              ) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, float]]:
    """
    Execute the baseline pipeline without visualization and return the estimated pose plus stats.
    Used by comparison scripts so the baseline can run inside the same Python process.
    """
    if registration_runs <= 0:
        raise ValueError("registration_runs must be positive")

    print("Reading model ...")
    cloud_model = o3d.io.read_point_cloud(model_path)
    if len(cloud_model.points) == 0:
        raise ValueError(f"Model cloud is empty: {model_path}")

    print("Training model ...")
    t_model_start = time.perf_counter()
    cloud_model_input = subsample_and_calculate_normals_model(
        cloud_model,
        voxel_size=sampling_leaf,
        k=5
    )
    angle_step = math.radians(angle_step_deg)
    distance_step = 0.6 * sampling_leaf
    ppf_model = build_ppf_model(cloud_model_input, angle_step, distance_step)
    t_model_end = time.perf_counter()
    model_build_time = t_model_end - t_model_start

    print("Reading scene ...")
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Scene cloud is empty: {scene_path}")
    print(f"Loaded {len(cloud_scene.points)} points.")

    rot_thresh_rad = math.radians(rot_thresh_deg)
    reg_durations: List[float] = []
    cloud_output: Optional[o3d.geometry.PointCloud] = None
    T_final = np.eye(4, dtype=float)

    for run_idx in range(registration_runs):
        t_run_start = time.perf_counter()
        # 场景使用普通法向估计，不做 model 侧的翻转逻辑
        cloud_scene_input = subsample_and_calculate_normals_scene(
            cloud_scene, voxel_size=sampling_leaf, k=5
        )

        cloud_output, T_final = ppf_register(
            ppf_model,
            cloud_scene_input,
            scene_ref_sampling_rate=scene_ref_sampling_rate,
            pos_thresh=pos_thresh,
            rot_thresh_rad=rot_thresh_rad
        )
        t_run_end = time.perf_counter()
        duration = t_run_end - t_run_start
        reg_durations.append(duration)
        print(f"一次场景注册时间 = {duration:.6f} sec (run {run_idx + 1}/{registration_runs})")

    avg_registration_time = sum(reg_durations) / len(reg_durations)
    stats = {
        "model_points": len(cloud_model.points),
        "scene_points": len(cloud_scene.points),
        "model_build_time": model_build_time,
        "avg_registration_time": avg_registration_time,
        "matching_rate": matching_rate(cloud_model, cloud_output, radius=5.0)
    }

    return T_final, cloud_output, stats


# ------------------------------------ Main ------------------------------------


def main(model_path: str = "dragon_m.ply", scene_path: str = "dragon_s.pcd"):
    sampling_leaf = 5.0  # same as original
    T_final, cloud_output, stats = run_baseline_registration(
        model_path=model_path,
        scene_path=scene_path,
        sampling_leaf=sampling_leaf,
        angle_step_deg=12.0,
        scene_ref_sampling_rate=20,
        pos_thresh=0.005,
        rot_thresh_deg=30.0,
        registration_runs=10
    )

    print(f"模型处理时间 = {stats['model_build_time']:.6f} sec")
    print(f"平均场景注册时间 = {stats['avg_registration_time']:.6f} sec")
    print(f"matching_rate = {stats['matching_rate']:.2f}%")

    # ---------------- Visualization (scene + registered model) ----------------
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PPF Object Recognition - Results",
                      width=1280, height=720)
    scene_draw = o3d.geometry.PointCloud(cloud_scene)
    vis.add_geometry(scene_draw)
    vis.poll_events()
    vis.update_renderer()

    model_draw = o3d.geometry.PointCloud(cloud_output)
    model_draw.paint_uniform_color(
        [random.random(), random.random(), random.random()])
    vis.add_geometry(model_draw)
    print("All models have been registered!")
    while True:
        vis.poll_events()
        vis.update_renderer()


if __name__ == "__main__":
    # Default behavior mirrors the C++: model == scene == 'dragon.ply'
    main()
