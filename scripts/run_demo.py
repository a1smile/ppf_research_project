import os
import sys
import argparse
import logging
import numpy as np
import open3d as o3d

# make project importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir
from ppf.registration import run_registration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--log_name", type=str, default="demo")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)

    out_logs = cfg["output"]["logs_dir"]
    ensure_dir(out_logs)
    log_path = os.path.join(out_logs, f"{args.log_name}.log")
    logger = setup_logger(log_path, level=logging.INFO)

    logger.info("=== RUN DEMO ===")
    logger.info(f"config={args.config}")
    logger.info(f"model={args.model}")
    logger.info(f"scene={args.scene}")
    logger.info(f"enable_rsmrq={cfg.get('enable_rsmrq', False)} enable_robust_vote={cfg.get('enable_robust_vote', False)} enable_kde_refine={cfg.get('enable_kde_refine', False)}")
    logger.info(f"full cfg:\n{cfg}")

    T_pred, out_model, debug, stats = run_registration(args.model, args.scene, cfg, logger=logger)

    logger.info(f"T_pred=\n{T_pred}")
    logger.info(f"stats={stats}")
    logger.info(f"debug={debug}")

    if bool(cfg.get("visualize", False)):
        scene = o3d.io.read_point_cloud(args.scene)
        model_vis = o3d.geometry.PointCloud(out_model)
        model_vis.paint_uniform_color([1.0, 0.2, 0.2])
        o3d.visualization.draw_geometries([scene, model_vis])


if __name__ == "__main__":
    main()
