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
    visibility: float = 0.00


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
class VisibilityCfg:
    enable: bool = False
    radius: float = 10.0
    normal_dot_thresh: float = 0.10
    require_normal_agreement: bool = False
    scene_normal_dot_thresh: float = 0.00


@dataclass
class CandidateVetoCfg:
    enable: bool = False

    # 绝对阈值：只有“同时很差”才会触发
    min_visibility_and_inlier_visibility: float = 0.12
    min_visibility_and_inlier_inlier: float = 0.22
    min_visibility_and_coverage_visibility: float = 0.10
    min_visibility_and_coverage_coverage: float = 0.30

    # 相对阈值：相对当前预选候选的最优值做自适应裁剪
    relative_visibility_ratio: float = 0.55
    relative_inlier_ratio: float = 0.60
    relative_coverage_ratio: float = 0.60

    # 安全项：无论如何至少保留这么多候选进入后续流程
    min_keep_candidates: int = 3


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
    visibility_support: float
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
            "visibility_support": float(self.visibility_support),
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


def _parse_pose_selection_cfg(
    cfg: Dict[str, Any]
) -> Tuple[PoseSelectionCfg, PoseScoreWeights, LightRefineCfg, VisibilityCfg, CandidateVetoCfg]:
    pose_cfg = cfg.get("pose_selection", {})
    weights_cfg = pose_cfg.get("weights", {})
    refine_cfg = pose_cfg.get("light_refine", {})
    visibility_cfg = pose_cfg.get("visibility", {})
    veto_cfg = pose_cfg.get("candidate_veto", {})

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
        visibility=float(weights_cfg.get("visibility", 0.00)),
    )
    light_refine = LightRefineCfg(
        enable=bool(refine_cfg.get("enable", True)),
        max_iter=int(refine_cfg.get("max_iter", 5)),
        distance_threshold=float(refine_cfg.get("distance_threshold", 8.0)),
    )
    vis_cfg = VisibilityCfg(
        enable=bool(visibility_cfg.get("enable", False)),
        radius=float(visibility_cfg.get("radius", pose_cfg.get("inlier_radius", 5.0))),
        normal_dot_thresh=float(visibility_cfg.get("normal_dot_thresh", 0.10)),
        require_normal_agreement=bool(visibility_cfg.get("require_normal_agreement", False)),
        scene_normal_dot_thresh=float(visibility_cfg.get("scene_normal_dot_thresh", 0.00)),
    )
    candidate_veto = CandidateVetoCfg(
        enable=bool(veto_cfg.get("enable", False)),
        min_visibility_and_inlier_visibility=float(veto_cfg.get("min_visibility_and_inlier_visibility", 0.12)),
        min_visibility_and_inlier_inlier=float(veto_cfg.get("min_visibility_and_inlier_inlier", 0.22)),
        min_visibility_and_coverage_visibility=float(veto_cfg.get("min_visibility_and_coverage_visibility", 0.10)),
        min_visibility_and_coverage_coverage=float(veto_cfg.get("min_visibility_and_coverage_coverage", 0.30)),
        relative_visibility_ratio=float(veto_cfg.get("relative_visibility_ratio", 0.55)),
        relative_inlier_ratio=float(veto_cfg.get("relative_inlier_ratio", 0.60)),
        relative_coverage_ratio=float(veto_cfg.get("relative_coverage_ratio", 0.60)),
        min_keep_candidates=int(veto_cfg.get("min_keep_candidates", 3)),
    )
    return ps_cfg, weights, light_refine, vis_cfg, candidate_veto


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


def _compute_visibility_support(
    T: np.ndarray,
    model_pts: np.ndarray,
    model_normals: np.ndarray,
    scene_pts: np.ndarray,
    scene_normals: np.ndarray,
    scene_kd: o3d.geometry.KDTreeFlann,
    vis_cfg: VisibilityCfg,
) -> float:
    """
    轻量版可见支持率：
    1) 把模型点变换到当前候选姿态下
    2) 只保留法向朝向相机的“前向面”点（相机在原点）
    3) 统计这些理论可见点中，有多少能在场景点云中找到近邻支持
    """
    if not vis_cfg.enable:
        return 0.0
    if model_pts.shape[0] == 0:
        return 0.0
    if model_normals.shape != model_pts.shape or model_normals.shape[0] == 0:
        return 0.0
    if scene_pts.shape[0] == 0:
        return 0.0

    transformed_pts = transform_points(T, model_pts)
    R = T[:3, :3]
    transformed_normals = (R @ model_normals.T).T

    view_dirs = -transformed_pts
    view_norm = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    valid_view = view_norm[:, 0] > 1e-12
    if not np.any(valid_view):
        return 0.0

    view_dirs = view_dirs / (view_norm + 1e-12)
    facing = np.sum(transformed_normals * view_dirs, axis=1) > float(vis_cfg.normal_dot_thresh)
    visible_mask = valid_view & facing
    visible_idx = np.where(visible_mask)[0]
    if visible_idx.shape[0] == 0:
        return 0.0

    radius2 = float(vis_cfg.radius) * float(vis_cfg.radius)
    supported = 0
    total_visible = 0

    has_scene_normals = scene_normals.shape == scene_pts.shape and scene_normals.shape[0] > 0

    for i in visible_idx:
        total_visible += 1
        _, idxs, d2 = scene_kd.search_knn_vector_3d(transformed_pts[i], 1)
        if not idxs or not d2:
            continue
        if float(d2[0]) > radius2:
            continue

        if vis_cfg.require_normal_agreement and has_scene_normals:
            dot = float(np.dot(transformed_normals[i], scene_normals[idxs[0]]))
            dot = max(-1.0, min(1.0, dot))
            if dot < float(vis_cfg.scene_normal_dot_thresh):
                continue

        supported += 1

    return float(supported) / max(1, total_visible)


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
) -> Tuple[float, float, float, float, float]:
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


def _absolute_veto_triggered(cand: PoseEvalResult, veto_cfg: CandidateVetoCfg) -> bool:
    # 规则 1：可见支持很低，同时 inlier 也偏低 -> 触发 veto
    if (
        cand.visibility_support < veto_cfg.min_visibility_and_inlier_visibility
        and cand.inlier_ratio < veto_cfg.min_visibility_and_inlier_inlier
    ):
        return True

    # 规则 2：可见支持极低，同时覆盖率也偏低 -> 触发 veto
    if (
        cand.visibility_support < veto_cfg.min_visibility_and_coverage_visibility
        and cand.coverage < veto_cfg.min_visibility_and_coverage_coverage
    ):
        return True

    return False


def _relative_veto_triggered(
    cand: PoseEvalResult,
    max_visibility_support: float,
    max_inlier_ratio: float,
    max_coverage: float,
    veto_cfg: CandidateVetoCfg,
) -> bool:
    vis_thresh = veto_cfg.relative_visibility_ratio * max(1e-12, max_visibility_support)
    inlier_thresh = veto_cfg.relative_inlier_ratio * max(1e-12, max_inlier_ratio)
    cov_thresh = veto_cfg.relative_coverage_ratio * max(1e-12, max_coverage)

    low_vis = cand.visibility_support < vis_thresh
    low_inlier = cand.inlier_ratio < inlier_thresh
    low_cov = cand.coverage < cov_thresh

    # 相对 veto 也要求“至少两个维度同时落后”，避免误杀
    return (low_vis and low_inlier) or (low_vis and low_cov)


def _apply_candidate_veto(
    candidates: List[PoseEvalResult],
    veto_cfg: CandidateVetoCfg,
) -> Tuple[List[PoseEvalResult], List[PoseEvalResult], bool]:
    """
    返回：
    - 保留候选
    - 被 veto 的候选
    - 是否触发安全回退
    """
    if not veto_cfg.enable or len(candidates) == 0:
        return candidates, [], False

    max_visibility_support = max(float(c.visibility_support) for c in candidates)
    max_inlier_ratio = max(float(c.inlier_ratio) for c in candidates)
    max_coverage = max(float(c.coverage) for c in candidates)

    kept: List[PoseEvalResult] = []
    vetoed: List[PoseEvalResult] = []

    for cand in candidates:
        abs_bad = _absolute_veto_triggered(cand, veto_cfg)
        rel_bad = _relative_veto_triggered(
            cand,
            max_visibility_support=max_visibility_support,
            max_inlier_ratio=max_inlier_ratio,
            max_coverage=max_coverage,
            veto_cfg=veto_cfg,
        )

        # 只有绝对差 + 相对也差时才 veto，避免 hardest cases 被一刀切
        if abs_bad and rel_bad:
            vetoed.append(cand)
        else:
            kept.append(cand)

    # 安全回退 1：如果全被 veto，就全量保留
    if len(kept) == 0:
        return candidates, [], True

    # 安全回退 2：如果保留下来的候选太少，就按原 score 从 vetoed 里补回一些
    min_keep = max(1, int(veto_cfg.min_keep_candidates))
    if len(kept) < min_keep and len(vetoed) > 0:
        vetoed_sorted = sorted(
            vetoed,
            key=lambda x: (
                x.score,
                x.inlier_ratio,
                x.coverage,
                x.visibility_support,
                x.vote,
            ),
            reverse=True,
        )
        need = min_keep - len(kept)
        kept.extend(vetoed_sorted[:need])
        vetoed = vetoed_sorted[need:]

    kept.sort(
        key=lambda x: (
            x.score,
            x.inlier_ratio,
            x.coverage,
            x.visibility_support,
            x.vote,
        ),
        reverse=True,
    )
    vetoed.sort(
        key=lambda x: (
            x.score,
            x.inlier_ratio,
            x.coverage,
            x.visibility_support,
            x.vote,
        ),
        reverse=True,
    )
    return kept, vetoed, False


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
    vis_cfg: VisibilityCfg,
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

    visibility_support = _compute_visibility_support(
        T=T,
        model_pts=model_pts,
        model_normals=model_normals,
        scene_pts=scene_pts,
        scene_normals=scene_normals,
        scene_kd=scene_kd,
        vis_cfg=vis_cfg,
    )

    score = (
        weights.vote * vote_norm
        + weights.inlier * inlier_ratio
        + weights.coverage * coverage
        + weights.normal * normal_consistency
        + weights.residual * residual_score
        + weights.visibility * visibility_support
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
        visibility_support=float(visibility_support),
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
    ps_cfg, weights, light_refine_cfg, vis_cfg, veto_cfg = _parse_pose_selection_cfg(cfg)

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
            vis_cfg=vis_cfg,
            refined=False,
        )
        evaluated.append(result)

    evaluated.sort(
        key=lambda x: (
            x.score,
            x.inlier_ratio,
            x.coverage,
            x.visibility_support,
            x.vote,
        ),
        reverse=True,
    )

    filtered, vetoed_candidates, veto_fallback_used = _apply_candidate_veto(
        evaluated,
        veto_cfg=veto_cfg,
    )
    veto_rejected = len(vetoed_candidates)

    selected = filtered[: min(ps_cfg.candidate_top_k, len(filtered))]

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
                vis_cfg=vis_cfg,
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
                        f"vis={cand.visibility_support:.4f}->{refined_eval.visibility_support:.4f} "
                        f"res={cand.residual_mean:.4f}->{refined_eval.residual_mean:.4f}"
                    )
            elif logger:
                logger.info(
                    "[PoseSelection][LightRefine] "
                    f"idx={cand.source_index} kept_original score={cand.score:.4f} refined={refined_eval.score:.4f}"
                )

        selected.sort(
            key=lambda x: (
                x.score,
                x.inlier_ratio,
                x.coverage,
                x.visibility_support,
                x.vote,
            ),
            reverse=True,
        )

        selected_after_refine, vetoed_after_refine, veto_fallback_after_refine = _apply_candidate_veto(
            selected,
            veto_cfg=veto_cfg,
        )
        if len(selected_after_refine) > 0:
            selected = selected_after_refine
            veto_rejected += len(vetoed_after_refine)
            veto_fallback_used = bool(veto_fallback_used or veto_fallback_after_refine)

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
        "num_veto_rejected": int(veto_rejected),
        "veto_fallback_used": bool(veto_fallback_used),
        "score_weights": {
            "vote": float(weights.vote),
            "inlier": float(weights.inlier),
            "coverage": float(weights.coverage),
            "normal": float(weights.normal),
            "residual": float(weights.residual),
            "visibility": float(weights.visibility),
        },
        "visibility_cfg": {
            "enable": bool(vis_cfg.enable),
            "radius": float(vis_cfg.radius),
            "normal_dot_thresh": float(vis_cfg.normal_dot_thresh),
            "require_normal_agreement": bool(vis_cfg.require_normal_agreement),
            "scene_normal_dot_thresh": float(vis_cfg.scene_normal_dot_thresh),
        },
        "candidate_veto_cfg": {
            "enable": bool(veto_cfg.enable),
            "min_visibility_and_inlier_visibility": float(veto_cfg.min_visibility_and_inlier_visibility),
            "min_visibility_and_inlier_inlier": float(veto_cfg.min_visibility_and_inlier_inlier),
            "min_visibility_and_coverage_visibility": float(veto_cfg.min_visibility_and_coverage_visibility),
            "min_visibility_and_coverage_coverage": float(veto_cfg.min_visibility_and_coverage_coverage),
            "relative_visibility_ratio": float(veto_cfg.relative_visibility_ratio),
            "relative_inlier_ratio": float(veto_cfg.relative_inlier_ratio),
            "relative_coverage_ratio": float(veto_cfg.relative_coverage_ratio),
            "min_keep_candidates": int(veto_cfg.min_keep_candidates),
        },
    }

    if vetoed_candidates:
        debug["top_vetoed_visibility_supports"] = [float(c.visibility_support) for c in vetoed_candidates[:5]]
        debug["top_vetoed_inlier_ratios"] = [float(c.inlier_ratio) for c in vetoed_candidates[:5]]
        debug["top_vetoed_coverages"] = [float(c.coverage) for c in vetoed_candidates[:5]]

    if selected:
        debug["best_score"] = float(selected[0].score)
        debug["best_vote"] = float(selected[0].vote)
        debug["best_inlier_ratio"] = float(selected[0].inlier_ratio)
        debug["best_coverage"] = float(selected[0].coverage)
        debug["best_normal_consistency"] = float(selected[0].normal_consistency)
        debug["best_residual_mean"] = float(selected[0].residual_mean)
        debug["best_visibility_support"] = float(selected[0].visibility_support)
        debug["top_scores"] = [float(c.score) for c in selected[:5]]
        debug["top_votes"] = [float(c.vote) for c in selected[:5]]
        debug["top_inlier_ratios"] = [float(c.inlier_ratio) for c in selected[:5]]
        debug["top_coverages"] = [float(c.coverage) for c in selected[:5]]
        debug["top_visibility_supports"] = [float(c.visibility_support) for c in selected[:5]]
        debug["top_sources"] = [str(c.source_stage) for c in selected[:5]]
    else:
        debug["best_score"] = 0.0

    return hypotheses, debug
