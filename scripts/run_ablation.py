import os
import sys
import argparse
import logging
import glob
import math
import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir, save_json
from ppf.registration import run_registration
from ppf.metrics import compute_metrics


def method_variants(base_cfg: dict):
    """
    Must run:
    - Baseline (all off)
    - +RS-MRQ
    - +RobustVote
    - +KDE
    - Full (all on)
    """
    def clone():
        import copy
        return copy.deepcopy(base_cfg)

    variants = []

    # Baseline
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = False
    variants.append(("baseline", c))

    # +RS-MRQ
    c = clone()
    c["enable_rsmrq"] = True
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = False
    variants.append(("plus_rsmrq", c))

    # +RobustVote
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = True
    c["enable_kde_refine"] = False
    variants.append(("plus_robustvote", c))

    # +KDE
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = True
    variants.append(("plus_kde", c))

    # Full
    c = clone()
    c["enable_rsmrq"] = True
    c["enable_robust_vote"] = True
    c["enable_kde_refine"] = True
    variants.append(("full", c))

    return variants


def load_optional_gt(gt_path: str):
    if not gt_path:
        return None
    import json
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    T = np.array(data["T"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("GT pose must be 4x4 in json field 'T'")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--gt_pose", type=str, default="", help="optional json with {'T':[[...4x4...]]}")
    ap.add_argument("--scan_ablation_configs", action="store_true", help="also scan configs/ablation_*.yaml and run them")
    args = ap.parse_args()

    cfg_path = os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config
    base_cfg = load_config(cfg_path)

    out_res = base_cfg["output"]["results_dir"]
    out_logs = base_cfg["output"]["logs_dir"]
    ensure_dir(out_res)
    ensure_dir(out_logs)

    T_gt = load_optional_gt(args.gt_pose)

    # collect methods
    methods = []
    # required 5 methods
    methods.extend(method_variants(base_cfg))

    # optional: scan ablation configs
    if args.scan_ablation_configs:
        for y in sorted(glob.glob(os.path.join(ROOT, "configs", "ablation_*.yaml"))):
            name = os.path.splitext(os.path.basename(y))[0]
            c = load_config(y)
            methods.append((name, c))

    # run
    for method_name, cfg in methods:
        for run_id in range(args.repeat):
            seed = int(cfg.get("seed", 0)) + run_id
            cfg["seed"] = seed

            log_path = os.path.join(out_logs, f"{method_name}_run{run_id}.log")
            logger = setup_logger(log_path, level=logging.INFO)

            logger.info("=== RUN ABLATION ===")
            logger.info(f"method={method_name} run_id={run_id} seed={seed}")
            logger.info(f"model={args.model}")
            logger.info(f"scene={args.scene}")
            logger.info(f"enable_rsmrq={cfg.get('enable_rsmrq', False)} enable_robust_vote={cfg.get('enable_robust_vote', False)} enable_kde_refine={cfg.get('enable_kde_refine', False)}")
            logger.info(f"cfg:\n{cfg}")

            T_pred, out_model, debug, stats = run_registration(args.model, args.scene, cfg, logger=logger)

            model_pts = np.asarray(o3d.io.read_point_cloud(args.model).points, dtype=np.float64)
            scene_pts = np.asarray(o3d.io.read_point_cloud(args.scene).points, dtype=np.float64)

            metrics = compute_metrics(model_pts, scene_pts, T_pred, T_gt=T_gt, inlier_radius=5.0)

            rec = {
                "method": method_name,
                "run_id": int(run_id),
                "seed": int(seed),
                "model_path": args.model,
                "scene_path": args.scene,
                "enable_rsmrq": bool(cfg.get("enable_rsmrq", False)),
                "enable_robust_vote": bool(cfg.get("enable_robust_vote", False)),
                "enable_kde_refine": bool(cfg.get("enable_kde_refine", False)),
                "ADD": float(metrics.get("ADD", float("nan"))),
                "ADD_S": float(metrics.get("ADD_S", float("nan"))),
                "rotation_error_deg": float(metrics.get("rotation_error_deg", float("nan"))),
                "translation_error": float(metrics.get("translation_error", float("nan"))),
                "inlier_ratio": float(metrics.get("inlier_ratio", float("nan"))),
                "model_build_time": float(stats.model_build_time),
                "registration_time": float(stats.registration_time),
                "total_time": float(stats.total_time),
                "candidate_inflation_mean": float(stats.candidate_inflation_mean),
                "robust_vote_summary": stats.robust_vote_summary,
                "kde_refine_calls": int(stats.kde_refine_calls),
                "T_pred": T_pred.tolist()
            }

            out_json = os.path.join(out_res, f"{method_name}_run{run_id}.json")
            save_json(out_json, rec)
            logger.info(f"Saved result: {out_json}")

            # required logs: include key stats
            logger.info(f"[RS-MRQ] candidate_inflation_mean={stats.candidate_inflation_mean:.3f}")
            if cfg.get("enable_robust_vote", False):
                logger.info(f"[RobustVote] summary={stats.robust_vote_summary}")
            if cfg.get("enable_kde_refine", False):
                logger.info(f"[KDERefine] calls={stats.kde_refine_calls}")
            logger.info(f"[Timing] model_build={stats.model_build_time:.6f} reg={stats.registration_time:.6f} total={stats.total_time:.6f}")


if __name__ == "__main__":
    main()
