import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir, save_json
from ppf.registration import run_registration
from ppf.metrics import compute_metrics
from ppf.bop_gt import scene_dir_from_depth_path, try_get_bop_gt_pose


def find_model_path(models_dir: str, obj_id: int) -> str:
    """
    Try common BOP naming patterns.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ablation_full.yaml")
    ap.add_argument("--csv", type=str, required=True, help="CSV should contain at least: pcd_path, obj_token, frame_id, depth_path (for BOP GT)")
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="batch")
    ap.add_argument("--limit", type=int, default=-1)

    # NEW: BOP GT support
    ap.add_argument("--use_bop_gt", action="store_true", help="If set, read scene_gt.json to get true obj_id and T_gt.")
    ap.add_argument("--t_scale", type=float, default=1.0, help="GT translation scale: 1.0 for mm, 0.001 for meters.")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)

    ensure_dir(cfg["output"]["results_dir"])
    ensure_dir(cfg["output"]["logs_dir"])

    log_path = os.path.join(cfg["output"]["logs_dir"], f"{args.out_prefix}.log")
    logger = setup_logger(log_path, level=logging.INFO)

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)

    # Basic column checks
    need_cols = ["pcd_path"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    use_bop_gt = bool(args.use_bop_gt)
    if use_bop_gt:
        for c in ["obj_token", "frame_id", "depth_path"]:
            if c not in df.columns:
                raise ValueError(f"--use_bop_gt requires CSV column: {c}")

    results = []
    n_gt_ok = 0
    n_gt_fail = 0

    for idx, row in df.iterrows():
        scene_path = str(row["pcd_path"])
        if not os.path.exists(scene_path):
            logger.warning(f"[{idx}] scene pcd missing: {scene_path}")
            continue

        T_gt = None
        obj_id = None
        obj_id_src = "csv"

        # Prefer BOP GT (robust): obj_token is gt_id in that frame
        if use_bop_gt:
            depth_path = str(row["depth_path"])
            frame_id = int(row["frame_id"])
            gt_id = int(row["obj_token"])

            scene_dir = scene_dir_from_depth_path(depth_path)
            obj_id_gt, T_gt_tmp, err = try_get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=float(args.t_scale))
            if err is None and obj_id_gt is not None and T_gt_tmp is not None:
                obj_id = int(obj_id_gt)
                T_gt = T_gt_tmp
                obj_id_src = "bop_gt"
                n_gt_ok += 1
            else:
                n_gt_fail += 1
                logger.warning(f"[{idx}] BOP GT failed: {err}. Fallback to CSV obj_id if exists.")

        # Fallback to CSV obj_id
        if obj_id is None:
            if "obj_id" not in df.columns:
                raise ValueError("No obj_id in CSV and BOP GT not available. Provide obj_id or enable --use_bop_gt.")
            obj_id = int(row["obj_id"])

        model_path = find_model_path(args.models_dir, obj_id)

        logger.info(f"[{idx+1}/{len(df)}] obj_id={obj_id} (src={obj_id_src}) scene={scene_path}")

        T_pred, out_model, debug, stats = run_registration(model_path, scene_path, cfg, logger=logger)

        # Metrics
        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)
        m = compute_metrics(model_pts, scene_pts, T_pred, T_gt=T_gt, inlier_radius=5.0)

        rec = {
            "idx": int(idx),
            "obj_id": int(obj_id),
            "obj_id_src": obj_id_src,
            "model_path": model_path,
            "scene_path": scene_path,
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
            "debug": debug
        }
        results.append(rec)

    out_json = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_batch.json")
    save_json(out_json, {"results": results, "gt_ok": n_gt_ok, "gt_fail": n_gt_fail, "use_bop_gt": use_bop_gt, "t_scale": float(args.t_scale)})
    logger.info(f"Saved batch results to {out_json}")
    if use_bop_gt:
        logger.info(f"BOP GT: ok={n_gt_ok} fail={n_gt_fail} (t_scale={args.t_scale})")


if __name__ == "__main__":
    main()
