import os
import sys
import argparse
import logging
import traceback
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir, save_json
from ppf.registration import run_registration
from ppf.metrics import compute_metrics
from ppf.bop_gt import scene_dir_from_depth_path, try_get_bop_gt_pose


# =========================
# 全局变量（供 worker 进程使用）
# =========================
G_CFG = None
G_MODELS_DIR = None
G_USE_BOP_GT = False
G_T_SCALE = 1.0
G_RUN_DIR = None
G_LOGGER = None
G_METHOD_NAME = None


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


def init_worker(cfg, models_dir, use_bop_gt, t_scale, run_dir, logger, method_name):
    global G_CFG, G_MODELS_DIR, G_USE_BOP_GT, G_T_SCALE, G_RUN_DIR, G_LOGGER, G_METHOD_NAME
    G_CFG = cfg
    G_MODELS_DIR = models_dir
    G_USE_BOP_GT = use_bop_gt
    G_T_SCALE = t_scale
    G_RUN_DIR = run_dir
    G_LOGGER = logger
    G_METHOD_NAME = method_name


def process_one(task):
    idx = task["idx"]
    row = task["row"]
    result = {"ok": False, "idx": int(idx), "error": "", "traceback": ""}

    try:
        scene_path = str(row["pcd_path"])
        if not os.path.exists(scene_path):
            result["error"] = f"scene pcd missing: {scene_path}"
            return result

        T_gt = None
        obj_id = None
        obj_id_src = "csv"

        if G_USE_BOP_GT:
            depth_path = str(row["depth_path"])
            frame_id = int(row["frame_id"])
            gt_id = int(row["obj_token"])

            scene_dir = scene_dir_from_depth_path(depth_path)
            obj_id_gt, T_gt_tmp, err = try_get_bop_gt_pose(
                scene_dir,
                frame_id,
                gt_id,
                t_scale=float(G_T_SCALE),
            )

            if err is None and obj_id_gt is not None and T_gt_tmp is not None:
                obj_id = int(obj_id_gt)
                T_gt = T_gt_tmp
                obj_id_src = "bop_gt"

        if obj_id is None:
            if "obj_id" in row and not pd.isna(row["obj_id"]):
                obj_id = int(row["obj_id"])
            elif "expected_obj_id" in row and not pd.isna(row["expected_obj_id"]):
                obj_id = int(row["expected_obj_id"])
            else:
                raise ValueError("No obj_id / expected_obj_id in CSV and BOP GT not available.")

        model_path = find_model_path(G_MODELS_DIR, obj_id)

        T_pred, out_model, debug, stats = run_registration(
            model_path,
            scene_path,
            G_CFG,
            logger=G_LOGGER,
        )

        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)

        m = compute_metrics(model_pts, scene_pts, T_pred, T_gt=T_gt, inlier_radius=5.0)

        rec = {
            "method": G_METHOD_NAME,
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
            "debug": debug,
        }

        per_case_dir = os.path.join(G_RUN_DIR, "results", "per_case")
        ensure_dir(per_case_dir)
        per_case_file = os.path.join(per_case_dir, f"{idx}_{obj_id}_result.json")
        save_json(per_case_file, rec)

        return {"ok": True, "idx": int(idx), "obj_id": int(obj_id), "record": rec}

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result


def _method_groups(config_dir: str):
    groups = [
        ("baseline", os.path.join(config_dir, "ablation_baseline.yaml")),
        ("rsmrq", os.path.join(config_dir, "ablation_rsmrq.yaml")),
        ("robustvote", os.path.join(config_dir, "ablation_robustvote.yaml")),
        ("rsmrq_robustvote", os.path.join(config_dir, "ablation_rsmrq_robustvote.yaml")),
        ("ours_no_rsmrq", os.path.join(config_dir, "ablation_ours_no_rsmrq.yaml")),
        ("ours_no_robustvote", os.path.join(config_dir, "ablation_ours_no_robustvote.yaml")),
        ("ours", os.path.join(config_dir, "ablation_ours.yaml")),
    ]
    return groups


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True,
                    help="CSV should contain at least: pcd_path, obj_token, frame_id, depth_path (for BOP GT)")
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--config_dir", type=str, default="configs",
                    help="Directory containing the 7 ablation YAML files.")
    ap.add_argument("--out_prefix", type=str, default="ablation")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--use_bop_gt", action="store_true",
                    help="If set, read scene_gt.json to get true obj_id and T_gt.")
    ap.add_argument("--t_scale", type=float, default=1.0,
                    help="GT translation scale: 1.0 for mm, 0.001 for meters.")
    ap.add_argument("--num_workers", type=int, default=16,
                    help="Number of worker processes.")
    ap.add_argument("--groups", type=str, default="",
                    help="Optional comma-separated subset of groups to run. "
                         "Choices: baseline,rsmrq,robustvote,rsmrq_robustvote,ours_no_rsmrq,ours_no_robustvote,ours")
    args = ap.parse_args()

    config_dir = _resolve_path(args.config_dir)
    models_dir = _resolve_path(args.models_dir)
    csv_path = _resolve_path(args.csv)

    all_groups = _method_groups(config_dir)
    group_names = [g[0] for g in all_groups]

    if args.groups.strip():
        wanted = [x.strip() for x in args.groups.split(",") if x.strip()]
        unknown = [x for x in wanted if x not in group_names]
        if unknown:
            raise ValueError(f"Unknown groups: {unknown}. Valid groups: {group_names}")
        methods = [(name, path) for name, path in all_groups if name in wanted]
    else:
        methods = all_groups

    for name, cfg_path in methods:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing ablation config for group '{name}': {cfg_path}")

    # 用第一份配置决定输出根目录
    first_cfg = load_config(methods[0][1])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(first_cfg["output"]["results_dir"], f"{args.out_prefix}_{ts}")
    ensure_dir(run_root)

    master_logs_dir = os.path.join(run_root, "logs")
    ensure_dir(master_logs_dir)
    master_log_path = os.path.join(master_logs_dir, f"{args.out_prefix}.log")
    master_logger = setup_logger(master_log_path, level=logging.INFO)

    df = pd.read_csv(csv_path)
    if args.limit > 0:
        df = df.head(args.limit)

    need_cols = ["pcd_path"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    use_bop_gt = bool(args.use_bop_gt)
    if use_bop_gt:
        for c in ["obj_token", "frame_id", "depth_path"]:
            if c not in df.columns:
                raise ValueError(f"--use_bop_gt requires CSV column: {c}")

    tasks = []
    for idx, row in df.iterrows():
        tasks.append({"idx": int(idx), "row": row.to_dict()})

    total_tasks = len(tasks)
    num_workers = int(args.num_workers)

    master_logger.info(f"CSV: {csv_path}")
    master_logger.info(f"Models dir: {models_dir}")
    master_logger.info(f"Total tasks: {total_tasks}")
    master_logger.info(f"Num workers: {num_workers}")
    master_logger.info(f"Groups: {[m[0] for m in methods]}")

    master_summary = {
        "csv": csv_path,
        "models_dir": models_dir,
        "total_tasks": total_tasks,
        "num_workers": num_workers,
        "groups": [],
        "use_bop_gt": bool(use_bop_gt),
        "t_scale": float(args.t_scale),
        "master_log_path": master_log_path,
    }

    for method_name, cfg_path in methods:
        cfg = load_config(cfg_path)
        method_dir = os.path.join(run_root, method_name)
        ensure_dir(method_dir)
        method_logs_dir = os.path.join(method_dir, "logs")
        method_results_dir = os.path.join(method_dir, "results")
        ensure_dir(method_logs_dir)
        ensure_dir(method_results_dir)

        method_log_path = os.path.join(method_logs_dir, f"{method_name}.log")
        logger = setup_logger(method_log_path, level=logging.INFO)

        logger.info("=== RUN ABLATION GROUP ===")
        logger.info(f"method={method_name}")
        logger.info(f"config={cfg_path}")
        logger.info(f"csv={csv_path}")
        logger.info(f"models_dir={models_dir}")
        logger.info(f"use_bop_gt={use_bop_gt} t_scale={args.t_scale}")
        logger.info(f"cfg:\n{cfg}")

        results = []
        failures = []
        n_gt_ok = 0
        n_gt_fail = 0

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(cfg, models_dir, use_bop_gt, args.t_scale, method_dir, logger, method_name),
        ) as pool:
            for ret in tqdm(
                pool.imap_unordered(process_one, tasks),
                total=total_tasks,
                desc=f"Processing[{method_name}]",
                unit="task",
            ):
                if ret["ok"]:
                    results.append(ret["record"])
                    n_gt_ok += 1
                else:
                    failures.append(ret)
                    logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
                    if "traceback" in ret:
                        logger.error(ret["traceback"])
                    if use_bop_gt:
                        n_gt_fail += 1

        results.sort(key=lambda x: x["idx"])

        out_json = os.path.join(method_results_dir, f"{args.out_prefix}_{method_name}_batch.json")
        save_json(
            out_json,
            {
                "method": method_name,
                "config_path": cfg_path,
                "results": results,
                "gt_ok": n_gt_ok,
                "gt_fail": n_gt_fail,
                "use_bop_gt": use_bop_gt,
                "t_scale": float(args.t_scale),
            }
        )

        summary_json = os.path.join(method_dir, "summary.json")
        method_summary = {
            "method": method_name,
            "config_path": cfg_path,
            "total_tasks": total_tasks,
            "success_count": len(results),
            "failure_count": len(failures),
            "gt_ok": n_gt_ok,
            "gt_fail": n_gt_fail,
            "num_workers": num_workers,
            "result_json": out_json,
            "log_path": method_log_path,
        }
        save_json(summary_json, method_summary)

        master_summary["groups"].append(method_summary)

        logger.info(f"Saved batch results to {out_json}")
        logger.info(f"Saved summary to {summary_json}")
        if use_bop_gt:
            logger.info(f"BOP GT: ok={n_gt_ok} fail={n_gt_fail} (t_scale={args.t_scale})")

        print(f"\n[DONE][{method_name}] Run directory: {method_dir}")
        print(f"[DONE][{method_name}] Batch JSON: {out_json}")
        print(f"[DONE][{method_name}] Log file: {method_log_path}")

    master_summary_json = os.path.join(run_root, "ablation_master_summary.json")
    save_json(master_summary_json, master_summary)
    master_logger.info(f"Saved master summary to {master_summary_json}")
    print(f"\n[DONE] Ablation root: {run_root}")
    print(f"[DONE] Master summary: {master_summary_json}")


if __name__ == "__main__":
    main()
