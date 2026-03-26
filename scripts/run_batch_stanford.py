import os
import sys
import math
import json
import hashlib
import argparse
import logging
import traceback
from multiprocessing import Pool
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir, save_json
from ppf.registration import run_registration
from ppf.metrics import compute_metrics
from ppf.preprocess import (
    subsample_and_calculate_normals_model,
    adaptive_subsample_and_calculate_normals_model,
)
from ppf.model_builder import build_ppf_model
from ppf.model_cache_io import (
    save_ppf_model_cache,
    load_ppf_model_cache,
    make_cache_meta,
)

G_CFG = None
G_RUN_DIR = None
G_LOGGER = None
G_INLIER_RADIUS = 5.0


def resolve_repo_path(path_str: str) -> str:
    path_str = str(path_str)
    return path_str if os.path.isabs(path_str) else os.path.join(ROOT, path_str)


def _is_valid(v) -> bool:
    return v is not None and not pd.isna(v) and str(v).strip() != ""


def find_model_path(models_dir: str, obj_id: int) -> str:
    candidates = [
        os.path.join(models_dir, f"{obj_id}.ply"),
        os.path.join(models_dir, f"{obj_id:06d}.ply"),
        os.path.join(models_dir, f"obj_{obj_id:06d}.ply"),
        os.path.join(models_dir, f"obj_{obj_id}.ply"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Cannot find model for obj_id={obj_id} under {models_dir}")


def parse_xf_matrix(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.extend(float(x) for x in s.split())

    if len(vals) == 16:
        T = np.array(vals, dtype=np.float64).reshape(4, 4)
    elif len(vals) == 12:
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
    else:
        raise ValueError(f"Unsupported .xf format: {path}, got {len(vals)} scalars")
    return T


def resolve_model_from_row(row: dict, models_dir: str) -> Tuple[str, str, Optional[int]]:
    explicit_cols = ["model_path", "expected_model_path", "expected_model"]
    for c in explicit_cols:
        if c in row and _is_valid(row[c]):
            p = resolve_repo_path(str(row[c]))
            if os.path.exists(p):
                obj_id = None
                for id_col in ["obj_id", "expected_obj_id"]:
                    if id_col in row and _is_valid(row[id_col]):
                        obj_id = int(row[id_col])
                        break
                return p, c, obj_id

    if "model_name" in row and _is_valid(row["model_name"]) and models_dir:
        p = os.path.join(models_dir, str(row["model_name"]))
        if os.path.exists(p):
            obj_id = None
            for id_col in ["obj_id", "expected_obj_id"]:
                if id_col in row and _is_valid(row[id_col]):
                    obj_id = int(row[id_col])
                    break
            return p, "model_name", obj_id

    obj_id = None
    for id_col in ["obj_id", "expected_obj_id"]:
        if id_col in row and _is_valid(row[id_col]):
            obj_id = int(row[id_col])
            break

    if obj_id is None:
        raise ValueError("No usable model column found. Need expected_model_path/model_path/model_name or obj_id.")
    if not models_dir:
        raise ValueError("CSV did not provide explicit model path and --models_dir was not set.")

    return find_model_path(models_dir, obj_id), "models_dir+obj_id", obj_id


def model_identifier(model_path: str) -> str:
    abs_path = os.path.abspath(model_path)
    try:
        return os.path.relpath(abs_path, ROOT)
    except Exception:
        return abs_path


def cache_signature_payload(model_path: str, cfg: dict) -> dict:
    adaptive_apply_to = str(cfg.get("adaptive_downsample_apply_to", "scene")).lower()
    return {
        "model_id": model_identifier(model_path),
        "angle_step_deg": float(cfg.get("angle_step_deg", 12.0)),
        "sampling_leaf": float(cfg.get("sampling_leaf", 5.0)),
        "normal_k": int(cfg.get("normal_k", 5)),
        "distance_step_ratio": float(cfg.get("distance_step_ratio", 0.6)),
        "enable_rsmrq": bool(cfg.get("enable_rsmrq", False)),
        "enable_robust_vote": bool(cfg.get("enable_robust_vote", False)),
        "rsmrq": dict(cfg.get("rsmrq", {}) or {}),
        "robust_vote": dict(cfg.get("robust_vote", {}) or {}),
        "adaptive_downsample": bool(cfg.get("adaptive_downsample", False)),
        "adaptive_downsample_apply_to": adaptive_apply_to,
        "adaptive_downsample_cfg": dict(cfg.get("adaptive_downsample_cfg", {}) or {}),
    }


def derive_cache_path(model_path: str, cfg: dict, cache_dir: str) -> str:
    ensure_dir(cache_dir)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    payload = cache_signature_payload(model_path, cfg)
    sig = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{stem}__{sig}.pkl")


def build_or_load_model_cache(model_path: str, cache_path: str, cfg: dict, logger=None, rebuild: bool = False) -> str:
    if (not rebuild) and os.path.exists(cache_path):
        try:
            load_ppf_model_cache(cache_path, cfg=cfg, strict=True)
            if logger:
                logger.info(f"[ModelCache] reuse {cache_path}")
            return cache_path
        except Exception as e:
            if logger:
                logger.warning(f"[ModelCache] invalid cache, rebuilding: {cache_path} ({e})")

    cloud_model = o3d.io.read_point_cloud(model_path)
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")

    sampling_leaf = float(cfg.get("sampling_leaf", 5.0))
    normal_k = int(cfg.get("normal_k", 5))
    angle_step_deg = float(cfg.get("angle_step_deg", 12.0))
    angle_step = math.radians(angle_step_deg)
    distance_step = float(cfg.get("distance_step_ratio", 0.6)) * sampling_leaf

    adaptive_downsample = bool(cfg.get("adaptive_downsample", False))
    adaptive_apply_to = str(cfg.get("adaptive_downsample_apply_to", "scene")).lower()
    adaptive_cfg = cfg.get("adaptive_downsample_cfg", {})

    if adaptive_downsample and adaptive_apply_to in ("model", "both"):
        model_down, ds_info = adaptive_subsample_and_calculate_normals_model(
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
        ds_info = {
            "raw_points": len(cloud_model.points),
            "down_points": len(model_down.points),
            "voxel_used": sampling_leaf,
            "adaptive_enabled": False,
        }

    if logger:
        logger.info(f"[ModelCache] build model={model_path} downsample={ds_info}")

    ppf_model = build_ppf_model(model_down, angle_step, distance_step, cfg, logger=logger)
    meta = make_cache_meta(model_identifier(model_path), cfg)
    save_ppf_model_cache(cache_path, ppf_model, meta)
    if logger:
        logger.info(f"[ModelCache] saved {cache_path}")
    return cache_path


def init_worker(cfg, run_dir, logger, inlier_radius):
    global G_CFG, G_RUN_DIR, G_LOGGER, G_INLIER_RADIUS
    G_CFG = cfg
    G_RUN_DIR = run_dir
    G_LOGGER = logger
    G_INLIER_RADIUS = float(inlier_radius)


def process_one(task):
    idx = task["idx"]
    row = task["row"]
    result = {"ok": False, "idx": int(idx), "error": "", "traceback": ""}

    try:
        scene_path_raw = str(row["pcd_path"])
        scene_path = resolve_repo_path(scene_path_raw)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"scene pcd missing: {scene_path} (raw={scene_path_raw})")

        model_path = str(row["_resolved_model_path"])
        model_src = str(row.get("_resolved_model_src", "csv"))
        obj_id = None if not _is_valid(row.get("_resolved_obj_id")) else int(row.get("_resolved_obj_id"))
        model_cache_path = str(row["_model_cache_path"])

        T_gt = None
        gt_path = None
        if "gt_path" in row and _is_valid(row["gt_path"]):
            gt_path = resolve_repo_path(str(row["gt_path"]))
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"gt_path missing: {gt_path}")
            T_gt = parse_xf_matrix(gt_path)

        T_pred, out_model, debug, stats = run_registration(
            model_path,
            scene_path,
            G_CFG,
            logger=G_LOGGER,
            model_cache_path=model_cache_path,
            strict_cache=True,
        )

        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)
        m = compute_metrics(
            model_pts,
            scene_pts,
            T_pred,
            T_gt=T_gt,
            inlier_radius=float(G_INLIER_RADIUS),
        )

        rec = {
            "idx": int(idx),
            "obj_id": (None if obj_id is None else int(obj_id)),
            "scene_id": (None if not _is_valid(row.get("scene_id")) else int(row["scene_id"])),
            "scene_name": row.get("scene_name"),
            "scene_variant": row.get("scene_variant"),
            "model_path": model_path,
            "model_path_src": model_src,
            "model_cache_path": model_cache_path,
            "scene_path": scene_path,
            "scene_path_raw": scene_path_raw,
            "gt_path": gt_path,
            "config_path": row.get("config_path"),
            "gt_threshold": row.get("gt_threshold"),
            "T_pred": T_pred.tolist(),
            "T_gt": (T_gt.tolist() if T_gt is not None else None),
            "stats": {
                "model_build_time": float(stats.model_build_time),
                "registration_time": float(stats.registration_time),
                "total_time": float(stats.total_time),
                "candidate_inflation_mean": float(stats.candidate_inflation_mean),
                "kde_refine_calls": int(stats.kde_refine_calls),
            },
            "metrics": m,
            "debug": debug,
            "row": {k: v for k, v in row.items() if not str(k).startswith("_")},
        }

        per_case_dir = os.path.join(G_RUN_DIR, "results", "per_case")
        ensure_dir(per_case_dir)
        scene_tag = row.get("scene_name") or f"idx{idx}"
        variant_tag = row.get("scene_variant", "na")
        per_case_file = os.path.join(per_case_dir, f"{idx:04d}_{scene_tag}_{variant_tag}.json")
        save_json(per_case_file, rec)

        return {"ok": True, "idx": int(idx), "record": rec}

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result


def main():
    ap = argparse.ArgumentParser(description="Batch runner for Stanford CSV datasets with per-model cache reuse.")
    ap.add_argument("--config", type=str, default="configs/ablation_ours.yaml")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--models_dir", type=str, default="", help="Fallback model directory when CSV has no explicit model path.")
    ap.add_argument("--cache_dir", type=str, default="data/stanford_bunny_ppf/model_cache")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--out_prefix", type=str, default="stanford_batch_cached")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--inlier_radius", type=float, default=5.0)
    args = ap.parse_args()

    config_path = resolve_repo_path(args.config)
    csv_path = resolve_repo_path(args.csv)
    models_dir = resolve_repo_path(args.models_dir) if str(args.models_dir).strip() else ""
    cache_dir = resolve_repo_path(args.cache_dir)

    cfg = load_config(config_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_{ts}")
    ensure_dir(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    results_dir = os.path.join(run_dir, "results")
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    log_path = os.path.join(logs_dir, f"{args.out_prefix}.log")
    logger = setup_logger(log_path, level=logging.INFO)

    df = pd.read_csv(csv_path)
    if args.limit > 0:
        df = df.head(args.limit)

    if "pcd_path" not in df.columns:
        raise ValueError("CSV missing required column: pcd_path")

    rows = []
    unique_models = {}
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        model_path, model_src, obj_id = resolve_model_from_row(row_dict, models_dir)
        cache_path = derive_cache_path(model_path, cfg, cache_dir)
        row_dict["_resolved_model_path"] = model_path
        row_dict["_resolved_model_src"] = model_src
        row_dict["_resolved_obj_id"] = obj_id
        row_dict["_model_cache_path"] = cache_path
        rows.append({"idx": int(idx), "row": row_dict})
        unique_models[model_path] = cache_path

    logger.info(f"Config: {config_path}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Models dir: {models_dir or '[not used]'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Total tasks: {len(rows)}")
    logger.info(f"Unique models: {len(unique_models)}")
    logger.info(f"Num workers: {int(args.num_workers)}")
    logger.info(f"Inlier radius: {float(args.inlier_radius)}")

    built_cache_records = []
    for model_path, cache_path in tqdm(unique_models.items(), total=len(unique_models), desc="BuildCache", unit="model"):
        build_or_load_model_cache(model_path, cache_path, cfg, logger=logger, rebuild=bool(args.rebuild_cache))
        built_cache_records.append({"model_path": model_path, "cache_path": cache_path})

    results = []
    failures = []
    with Pool(
        processes=int(args.num_workers),
        initializer=init_worker,
        initargs=(cfg, run_dir, logger, args.inlier_radius),
    ) as pool:
        for ret in tqdm(pool.imap_unordered(process_one, rows), total=len(rows), desc="Processing", unit="task"):
            if ret["ok"]:
                results.append(ret["record"])
            else:
                failures.append(ret)
                logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
                if ret.get("traceback"):
                    logger.error(ret["traceback"])

    results.sort(key=lambda x: x["idx"])

    out_json = os.path.join(results_dir, f"{args.out_prefix}_batch.json")
    save_json(
        out_json,
        {
            "config_path": config_path,
            "csv_path": csv_path,
            "models_dir": models_dir,
            "cache_dir": cache_dir,
            "caches": built_cache_records,
            "results": results,
            "failure_count": len(failures),
            "inlier_radius": float(args.inlier_radius),
        },
    )

    summary_json = os.path.join(run_dir, "summary.json")
    save_json(
        summary_json,
        {
            "config_path": config_path,
            "csv_path": csv_path,
            "models_dir": models_dir,
            "cache_dir": cache_dir,
            "total_tasks": len(rows),
            "unique_models": len(unique_models),
            "success_count": len(results),
            "failure_count": len(failures),
            "num_workers": int(args.num_workers),
            "result_json": out_json,
            "log_path": log_path,
            "script": os.path.basename(__file__),
        },
    )

    print(f"\n[DONE] Run directory: {run_dir}")
    print(f"[DONE] Batch JSON: {out_json}")
    print(f"[DONE] Log file: {log_path}")


if __name__ == "__main__":
    main()
