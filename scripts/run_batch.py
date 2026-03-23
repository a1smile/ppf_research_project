import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import open3d as o3d
from multiprocessing import Pool, Manager, Lock
from tqdm import tqdm  # 进度条
from datetime import datetime
import traceback

# 获取项目根目录路径。
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

# 导入配置读取函数。
from ppf.io import load_config
# 导入日志、目录创建和 JSON 保存工具函数。
from ppf.utils import setup_logger, ensure_dir, save_json
# 导入配准主流程函数。
from ppf.registration import run_registration
# 导入指标计算函数。
from ppf.metrics import compute_metrics
# 导入 BOP 真值读取辅助函数。
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


# 根据 obj_id 在模型目录中查找模型文件。
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


# 初始化每个子进程的全局变量。
def init_worker(cfg, models_dir, use_bop_gt, t_scale, run_dir, logger):
    global G_CFG, G_MODELS_DIR, G_USE_BOP_GT, G_T_SCALE, G_RUN_DIR, G_LOGGER
    G_CFG = cfg
    G_MODELS_DIR = models_dir
    G_USE_BOP_GT = use_bop_gt
    G_T_SCALE = t_scale
    G_RUN_DIR = run_dir
    G_LOGGER = logger


# 单个样本处理函数（worker 进程真正执行的任务）。
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
            if "obj_id" not in row:
                raise ValueError("No obj_id in CSV and BOP GT not available. Provide obj_id or enable --use_bop_gt.")
            obj_id = int(row["obj_id"])

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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="configs/ablation_full.yaml")
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV should contain at least: pcd_path, obj_token, frame_id, depth_path (for BOP GT)")
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="batch")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--use_bop_gt", action="store_true", help="If set, read scene_gt.json to get true obj_id and T_gt.")
    ap.add_argument("--t_scale", type=float, default=1.0, help="GT translation scale: 1.0 for mm, 0.001 for meters.")
    ap.add_argument("--num_workers", type=int, default=6, help="Number of worker processes.")
    args = ap.parse_args()

    cfg = load_config(args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_{ts}")
    ensure_dir(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    ensure_dir(logs_dir)
    results_dir = os.path.join(run_dir, "results")
    ensure_dir(results_dir)

    log_path = os.path.join(logs_dir, f"{args.out_prefix}.log")
    logger = setup_logger(log_path, level=logging.INFO)

    df = pd.read_csv(args.csv)
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

    results = []
    failures = []
    n_gt_ok = 0
    n_gt_fail = 0

    total_tasks = len(tasks)
    num_workers = int(args.num_workers)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Num workers: {num_workers}")

    with Pool(processes=num_workers, initializer=init_worker,
              initargs=(cfg, args.models_dir, use_bop_gt, args.t_scale, run_dir, logger)) as pool:
        for ret in tqdm(pool.imap_unordered(process_one, tasks), total=total_tasks, desc="Processing", unit="task"):
            if ret["ok"]:
                results.append(ret["record"])
                n_gt_ok += 1
            else:
                failures.append(ret)
                logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
                if "traceback" in ret:
                    logger.error(ret["traceback"])

    results.sort(key=lambda x: x["idx"])

    out_json = os.path.join(results_dir, f"{args.out_prefix}_batch.json")
    save_json(out_json, {"results": results, "gt_ok": n_gt_ok, "gt_fail": n_gt_fail, "use_bop_gt": use_bop_gt,
                         "t_scale": float(args.t_scale)})

    summary_json = os.path.join(run_dir, "summary.json")
    save_json(
        summary_json,
        {
            "total_tasks": total_tasks,
            "success_count": len(results),
            "failure_count": len(failures),
            "gt_ok": n_gt_ok,
            "gt_fail": n_gt_fail,
            "num_workers": num_workers,
            "result_json": out_json,
            "log_path": log_path,
        }
    )

    logger.info(f"Saved batch results to {out_json}")
    logger.info(f"Saved summary to {summary_json}")

    if use_bop_gt:
        logger.info(f"BOP GT: ok={n_gt_ok} fail={n_gt_fail} (t_scale={args.t_scale})")

    print(f"\n[DONE] Run directory: {run_dir}")
    print(f"[DONE] Batch JSON: {out_json}")
    print(f"[DONE] Log file: {log_path}")


if __name__ == "__main__":
    main()