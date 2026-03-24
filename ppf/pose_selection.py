from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .pose_clustering import PoseHypothesis
from .refine_icp import refine_icp_point_to_point
from .utils import transform_points


@dataclass
class PoseScoreWeights:
    vote: float = 0.25
    inlier: float = 0.30
    coverage: float = 0.20
    normal: float = 0.10
    residual: float = 0.15


@dataclass
class PoseSelectionCfg:
    enable: bool = False
    pre_top_m_by_vote: int = 80
    candidate_top_k: int = 10
    refine_top_k: int = 3
    inlier_radius: float = 5.0
    max_correspondence_distance: float = 12.0
    residual_sigma: float = 5.0
    coverage_grid_size: int = 4
    normal_use_abs_dot: bool = False
    keep_original_if_refine_worse: bool = True


@dataclass
class LightRefineCfg:
    enable: bool = True
    max_iter: int = 5
    distance_threshold: float = 8.0


@dataclass
class PoseEvalResult:
    T: np.ndarray
    vote: float
    vote_norm: float
    inlier_ratio: float
    residual_mean: float
    residual_score: float
    coverage: float
    normal_consistency: float
    score: float
    refined: bool
    source_index: int
    source_stage: str

    def to_hypothesis(self) -> PoseHypothesis:
        meta = {
            "vote": float(self.vote),
            "vote_norm": float(self.vote_norm),
            "inlier_ratio": float(self.inlier_ratio),
            "residual_mean": float(self.residual_mean),
            "residual_score": float(self.residual_score),
            "coverage": float(self.coverage),
            "normal_consistency": float(self.normal_consistency),
            "refined": bool(self.refined),
            "source_index": int(self.source_index),
            "source_stage": str(self.source_stage),
        }
        return PoseHypothesis(T=self.T.copy(), score=float(self.score), meta=meta)


def _build_point_cloud(points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return pcd


def _parse_pose_selection_cfg(cfg: Dict[str, Any]) -> Tuple[PoseSelectionCfg, PoseScoreWeights, LightRefineCfg]:
    pose_cfg = cfg.get("pose_selection", {})
    weights_cfg = pose_cfg.get("weights", {})
    refine_cfg = pose_cfg.get("light_refine", {})

    ps_cfg = PoseSelectionCfg(
        enable=bool(pose_cfg.get("enable", False)),
        pre_top_m_by_vote=int(pose_cfg.get("pre_top_m_by_vote", 80)),
        candidate_top_k=int(pose_cfg.get("candidate_top_k", 10)),
        refine_top_k=int(pose_cfg.get("refine_top_k", 3)),
        inlier_radius=float(pose_cfg.get("inlier_radius", 5.0)),
        max_correspondence_distance=float(pose_cfg.get("max_correspondence_distance", 12.0)),
        residual_sigma=float(pose_cfg.get("residual_sigma", pose_cfg.get("inlier_radius", 5.0))),
        coverage_grid_size=int(pose_cfg.get("coverage_grid_size", 4)),
        normal_use_abs_dot=bool(pose_cfg.get("normal_use_abs_dot", False)),
        keep_original_if_refine_worse=bool(pose_cfg.get("keep_original_if_refine_worse", True)),
    )
    weights = PoseScoreWeights(
        vote=float(weights_cfg.get("vote", 0.25)),
        inlier=float(weights_cfg.get("inlier", 0.30)),
        coverage=float(weights_cfg.get("coverage", 0.20)),
        normal=float(weights_cfg.get("normal", 0.10)),
        residual=float(weights_cfg.get("residual", 0.15)),
    )
    light_refine = LightRefineCfg(
        enable=bool(refine_cfg.get("enable", True)),
        max_iter=int(refine_cfg.get("max_iter", 5)),
        distance_threshold=float(refine_cfg.get("distance_threshold", 8.0)),
    )
    return ps_cfg, weights, light_refine


def _build_model_region_ids(model_pts: np.ndarray, grid_size: int) -> Tuple[np.ndarray, int]:
    if model_pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32), 0

    grid_size = max(2, int(grid_size))
    mins = np.min(model_pts, axis=0)
    maxs = np.max(model_pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    normed = (model_pts - mins[None, :]) / spans[None, :]
    bins = np.floor(normed * float(grid_size)).astype(np.int32)
    bins = np.clip(bins, 0, grid_size - 1)
    keys = bins[:, 0] + grid_size * bins[:, 1] + (grid_size * grid_size) * bins[:, 2]
    total_regions = int(np.unique(keys).shape[0])
    return keys.astype(np.int32), total_regions


def _nearest_neighbor_metrics(
    T: np.ndarray,
    model_pts: np.ndarray,
    model_normals: np.ndarray,
    model_region_ids: np.ndarray,
    total_regions: int,
    scene_pts: np.ndarray,
    scene_normals: np.ndarray,
    scene_kd: o3d.geometry.KDTreeFlann,
    max_corr_dist: float,
    inlier_radius: float,
    residual_sigma: float,
    normal_use_abs_dot: bool,
) -> Tuple[float, float, float, float]:
    transformed_pts = transform_points(T, model_pts)
    R = T[:3, :3]
    transformed_normals = None
    if model_normals.shape == model_pts.shape and model_normals.shape[0] > 0:
        transformed_normals = (R @ model_normals.T).T

    residuals: List[float] = []
    inlier_ids: List[int] = []
    normal_scores: List[float] = []
    max_corr_dist2 = float(max_corr_dist) * float(max_corr_dist)

    for i in range(transformed_pts.shape[0]):
        _, idxs, d2 = scene_kd.search_knn_vector_3d(transformed_pts[i], 1)
        if not idxs or not d2:
            continue
        dist2 = float(d2[0])
        if dist2 > max_corr_dist2:
            continue

        dist = math.sqrt(max(0.0, dist2))
        residuals.append(dist)

        if dist <= inlier_radius:
            inlier_ids.append(i)
            if transformed_normals is not None and scene_normals.shape == scene_pts.shape and scene_normals.shape[0] > idxs[0]:
                dot = float(np.dot(transformed_normals[i], scene_normals[idxs[0]]))
                dot = max(-1.0, min(1.0, dot))
                if normal_use_abs_dot:
                    dot = abs(dot)
                else:
                    dot = max(0.0, dot)
                normal_scores.append(dot)

    if residuals:
        residual_mean = float(np.mean(np.asarray(residuals, dtype=np.float64)))
    else:
        residual_mean = float(max_corr_dist)

    inlier_ratio = float(len(inlier_ids)) / max(1, model_pts.shape[0])

    if len(inlier_ids) > 0 and total_regions > 0:
        covered = int(np.unique(model_region_ids[np.asarray(inlier_ids, dtype=np.int64)]).shape[0])
        coverage = float(covered) / float(total_regions)
    else:
        coverage = 0.0

    normal_consistency = float(np.mean(np.asarray(normal_scores, dtype=np.float64))) if normal_scores else 0.0
    residual_score = math.exp(-residual_mean / max(1e-6, residual_sigma))
    return inlier_ratio, residual_mean, coverage, normal_consistency, residual_score


def _evaluate_pose(
    T: np.ndarray,
    vote: float,
    vote_norm: float,
    source_index: int,
    stage: str,
    model_pts: np.ndarray,
    model_normals: np.ndarray,
    model_region_ids: np.ndarray,
    total_regions: int,
    scene_pts: np.ndarray,
    scene_normals: np.ndarray,
    scene_kd: o3d.geometry.KDTreeFlann,
    ps_cfg: PoseSelectionCfg,
    weights: PoseScoreWeights,
    refined: bool,
) -> PoseEvalResult:
    inlier_ratio, residual_mean, coverage, normal_consistency, residual_score = _nearest_neighbor_metrics(
        T=T,
        model_pts=model_pts,
        model_normals=model_normals,
        model_region_ids=model_region_ids,
        total_regions=total_regions,
        scene_pts=scene_pts,
        scene_normals=scene_normals,
        scene_kd=scene_kd,
        max_corr_dist=ps_cfg.max_correspondence_distance,
        inlier_radius=ps_cfg.inlier_radius,
        residual_sigma=ps_cfg.residual_sigma,
        normal_use_abs_dot=ps_cfg.normal_use_abs_dot,
    )

    score = (
        weights.vote * vote_norm
        + weights.inlier * inlier_ratio
        + weights.coverage * coverage
        + weights.normal * normal_consistency
        + weights.residual * residual_score
    )

    return PoseEvalResult(
        T=T.copy(),
        vote=float(vote),
        vote_norm=float(vote_norm),
        inlier_ratio=float(inlier_ratio),
        residual_mean=float(residual_mean),
        residual_score=float(residual_score),
        coverage=float(coverage),
        normal_consistency=float(normal_consistency),
        score=float(score),
        refined=bool(refined),
        source_index=int(source_index),
        source_stage=str(stage),
    )


def select_pose_hypotheses(
    voted_poses: List[Any],
    model_pts: np.ndarray,
    model_normals: np.ndarray,
    scene_pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[PoseHypothesis], Dict[str, Any]]:
    ps_cfg, weights, light_refine_cfg = _parse_pose_selection_cfg(cfg)

    if not ps_cfg.enable or len(voted_poses) == 0:
        return [], {
            "enabled": False,
            "num_input_candidates": len(voted_poses),
            "num_preselected": 0,
            "num_selected": 0,
            "num_light_refined": 0,
        }

    scene_pts = np.asarray(scene_pcd.points, dtype=np.float64)
    scene_normals = np.asarray(scene_pcd.normals, dtype=np.float64)
    scene_kd = o3d.geometry.KDTreeFlann(scene_pcd)
    model_region_ids, total_regions = _build_model_region_ids(model_pts, ps_cfg.coverage_grid_size)

    sorted_votes = sorted(
        [
            (idx, np.asarray(getattr(vp, "T"), dtype=np.float64), float(getattr(vp, "votes", 1.0)))
            for idx, vp in enumerate(voted_poses)
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    pre_top_m = min(ps_cfg.pre_top_m_by_vote, len(sorted_votes))
    preselected = sorted_votes[:pre_top_m]
    max_vote = max(1e-12, float(preselected[0][2])) if preselected else 1.0

    evaluated: List[PoseEvalResult] = []
    for src_idx, T, vote in preselected:
        vote_norm = float(vote) / max_vote
        result = _evaluate_pose(
            T=T,
            vote=vote,
            vote_norm=vote_norm,
            source_index=src_idx,
            stage="raw",
            model_pts=model_pts,
            model_normals=model_normals,
            model_region_ids=model_region_ids,
            total_regions=total_regions,
            scene_pts=scene_pts,
            scene_normals=scene_normals,
            scene_kd=scene_kd,
            ps_cfg=ps_cfg,
            weights=weights,
            refined=False,
        )
        evaluated.append(result)

    evaluated.sort(key=lambda x: (x.score, x.inlier_ratio, x.coverage, x.vote), reverse=True)
    selected = evaluated[: min(ps_cfg.candidate_top_k, len(evaluated))]

    num_light_refined = 0
    if light_refine_cfg.enable and light_refine_cfg.max_iter > 0 and len(selected) > 0:
        source_pcd = _build_point_cloud(model_pts, model_normals)
        refine_count = min(ps_cfg.refine_top_k, len(selected))
        for i in range(refine_count):
            cand = selected[i]
            T_ref = refine_icp_point_to_point(
                source=source_pcd,
                target=scene_pcd,
                init_T=cand.T,
                distance_threshold=light_refine_cfg.distance_threshold,
                max_iter=light_refine_cfg.max_iter,
            )
            refined_eval = _evaluate_pose(
                T=T_ref,
                vote=cand.vote,
                vote_norm=cand.vote_norm,
                source_index=cand.source_index,
                stage="light_refine",
                model_pts=model_pts,
                model_normals=model_normals,
                model_region_ids=model_region_ids,
                total_regions=total_regions,
                scene_pts=scene_pts,
                scene_normals=scene_normals,
                scene_kd=scene_kd,
                ps_cfg=ps_cfg,
                weights=weights,
                refined=True,
            )
            num_light_refined += 1

            if (not ps_cfg.keep_original_if_refine_worse) or (refined_eval.score >= cand.score):
                selected[i] = refined_eval
                if logger:
                    logger.info(
                        "[PoseSelection][LightRefine] "
                        f"idx={cand.source_index} score={cand.score:.4f}->{refined_eval.score:.4f} "
                        f"inlier={cand.inlier_ratio:.4f}->{refined_eval.inlier_ratio:.4f} "
                        f"res={cand.residual_mean:.4f}->{refined_eval.residual_mean:.4f}"
                    )
            elif logger:
                logger.info(
                    "[PoseSelection][LightRefine] "
                    f"idx={cand.source_index} kept_original score={cand.score:.4f} refined={refined_eval.score:.4f}"
                )

        selected.sort(key=lambda x: (x.score, x.inlier_ratio, x.coverage, x.vote), reverse=True)
        selected = selected[: min(ps_cfg.candidate_top_k, len(selected))]

    hypotheses = [cand.to_hypothesis() for cand in selected]

    debug: Dict[str, Any] = {
        "enabled": True,
        "num_input_candidates": len(voted_poses),
        "num_preselected": len(preselected),
        "num_selected": len(hypotheses),
        "num_light_refined": int(num_light_refined),
        "pre_top_m_by_vote": int(ps_cfg.pre_top_m_by_vote),
        "candidate_top_k": int(ps_cfg.candidate_top_k),
        "refine_top_k": int(ps_cfg.refine_top_k),
        "score_weights": {
            "vote": float(weights.vote),
            "inlier": float(weights.inlier),
            "coverage": float(weights.coverage),
            "normal": float(weights.normal),
            "residual": float(weights.residual),
        },
    }

    if selected:
        debug["best_score"] = float(selected[0].score)
        debug["best_vote"] = float(selected[0].vote)
        debug["best_inlier_ratio"] = float(selected[0].inlier_ratio)
        debug["best_coverage"] = float(selected[0].coverage)
        debug["best_normal_consistency"] = float(selected[0].normal_consistency)
        debug["best_residual_mean"] = float(selected[0].residual_mean)
        debug["top_scores"] = [float(c.score) for c in selected[:5]]
        debug["top_votes"] = [float(c.vote) for c in selected[:5]]
        debug["top_inlier_ratios"] = [float(c.inlier_ratio) for c in selected[:5]]
        debug["top_coverages"] = [float(c.coverage) for c in selected[:5]]
        debug["top_sources"] = [str(c.source_stage) for c in selected[:5]]
    else:
        debug["best_score"] = 0.0

    return hypotheses, debug
