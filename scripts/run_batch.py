# 导入 os 模块，用于路径处理和文件存在性判断。
import os
# 导入 sys 模块，用于修改模块搜索路径。
import sys
# 导入 argparse 模块，用于解析命令行参数。
import argparse
# 导入 logging 模块，用于日志输出。
import logging
# 导入 pandas 并命名为 pd，用于读取批处理 CSV。
import pandas as pd
# 导入 numpy 并命名为 np，用于点云数组转换。
import numpy as np
# 导入 open3d 并命名为 o3d，用于读取点云文件。
import open3d as o3d

# 获取项目根目录路径。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录加入 Python 模块搜索路径。
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


# 根据 obj_id 在模型目录中查找模型文件。
def find_model_path(models_dir: str, obj_id: int) -> str:
    """
    Try common BOP naming patterns.
    """
    # 列出常见的 BOP 模型命名方式作为候选路径。
    candidates = [
        os.path.join(models_dir, f"{obj_id}.ply"),
        os.path.join(models_dir, f"{obj_id:06d}.ply"),
        os.path.join(models_dir, f"obj_{obj_id:06d}.ply"),
        os.path.join(models_dir, f"obj_{obj_id}.ply"),
    ]
    # 依次检查候选路径是否存在。
    for c in candidates:
        # 找到第一个存在的路径后立即返回。
        if os.path.exists(c):
            return c
    # 如果所有候选路径都不存在，则抛出异常。
    raise FileNotFoundError(f"Cannot find model for obj_id={obj_id} under {models_dir}")


# 定义主函数，负责按 CSV 批量运行配准。
def main():
    # 创建命令行参数解析器。
    ap = argparse.ArgumentParser()
    # 添加配置文件路径参数。
    ap.add_argument("--config", type=str, default="configs/ablation_full.yaml")
    # 添加输入 CSV 路径参数。
    ap.add_argument("--csv", type=str, required=True, help="CSV should contain at least: pcd_path, obj_token, frame_id, depth_path (for BOP GT)")
    # 添加模型目录参数。
    ap.add_argument("--models_dir", type=str, required=True)
    # 添加输出前缀参数。
    ap.add_argument("--out_prefix", type=str, default="batch")
    # 添加处理条数上限参数。
    ap.add_argument("--limit", type=int, default=-1)

    # 添加是否使用 BOP 真值的开关参数。
    ap.add_argument("--use_bop_gt", action="store_true", help="If set, read scene_gt.json to get true obj_id and T_gt.")
    # 添加 GT 平移单位缩放参数。
    ap.add_argument("--t_scale", type=float, default=1.0, help="GT translation scale: 1.0 for mm, 0.001 for meters.")
    # 解析命令行参数。
    args = ap.parse_args()

    # 加载配置文件；若为相对路径则相对项目根目录解析。
    cfg = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)

    # 确保结果目录存在。
    ensure_dir(cfg["output"]["results_dir"])
    # 确保日志目录存在。
    ensure_dir(cfg["output"]["logs_dir"])

    # 构造本次批处理的日志路径。
    log_path = os.path.join(cfg["output"]["logs_dir"], f"{args.out_prefix}.log")
    # 创建日志记录器。
    logger = setup_logger(log_path, level=logging.INFO)

    # 读取输入 CSV。
    df = pd.read_csv(args.csv)
    # 如果设置了 limit，则只取前 limit 条记录。
    if args.limit > 0:
        df = df.head(args.limit)

    # 定义最基础必须存在的列。
    need_cols = ["pcd_path"]
    # 检查这些列是否存在于 CSV 中。
    for c in need_cols:
        # 缺列时直接报错。
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # 将开关参数转成显式布尔值。
    use_bop_gt = bool(args.use_bop_gt)
    # 如果启用了 BOP GT，则检查额外所需列。
    if use_bop_gt:
        # BOP GT 模式下这三列是必须的。
        for c in ["obj_token", "frame_id", "depth_path"]:
            # 若缺列则报错。
            if c not in df.columns:
                raise ValueError(f"--use_bop_gt requires CSV column: {c}")

    # 用于保存所有样本的结果。
    results = []
    # 记录 GT 成功读取次数。
    n_gt_ok = 0
    # 记录 GT 读取失败次数。
    n_gt_fail = 0

    # 逐行处理 CSV 中的样本。
    for idx, row in df.iterrows():
        # 读取场景点云路径。
        scene_path = str(row["pcd_path"])
        # 如果场景点云文件不存在，则记录警告并跳过。
        if not os.path.exists(scene_path):
            logger.warning(f"[{idx}] scene pcd missing: {scene_path}")
            continue

        # 初始化 GT 位姿为空。
        T_gt = None
        # 初始化 obj_id，稍后由 GT 或 CSV 提供。
        obj_id = None
        # 记录 obj_id 的来源，默认来自 CSV。
        obj_id_src = "csv"

        # 如果启用了 BOP GT，则优先使用真值读取 obj_id 和 GT 位姿。
        if use_bop_gt:
            # 读取深度图路径。
            depth_path = str(row["depth_path"])
            # 读取帧编号。
            frame_id = int(row["frame_id"])
            # 读取当前帧内的目标实例编号。
            gt_id = int(row["obj_token"])

            # 根据深度图路径推断场景目录。
            scene_dir = scene_dir_from_depth_path(depth_path)
            # 尝试从 BOP 真值中读取 obj_id 和 GT 位姿。
            obj_id_gt, T_gt_tmp, err = try_get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=float(args.t_scale))
            # 如果成功读取，则更新相关变量并统计成功次数。
            if err is None and obj_id_gt is not None and T_gt_tmp is not None:
                obj_id = int(obj_id_gt)
                T_gt = T_gt_tmp
                obj_id_src = "bop_gt"
                n_gt_ok += 1
            else:
                # 若读取失败，则统计失败次数并在日志中提示会回退到 CSV obj_id。
                n_gt_fail += 1
                logger.warning(f"[{idx}] BOP GT failed: {err}. Fallback to CSV obj_id if exists.")

        # 如果未从 BOP GT 获取到 obj_id，则回退到 CSV 的 obj_id 列。
        if obj_id is None:
            # 如果 CSV 也没有 obj_id，则无法继续处理。
            if "obj_id" not in df.columns:
                raise ValueError("No obj_id in CSV and BOP GT not available. Provide obj_id or enable --use_bop_gt.")
            # 从 CSV 中读取 obj_id。
            obj_id = int(row["obj_id"])

        # 根据 obj_id 找到对应模型文件路径。
        model_path = find_model_path(args.models_dir, obj_id)

        # 记录当前处理进度和关键信息。
        logger.info(f"[{idx+1}/{len(df)}] obj_id={obj_id} (src={obj_id_src}) scene={scene_path}")

        # 执行配准流程。
        T_pred, out_model, debug, stats = run_registration(model_path, scene_path, cfg, logger=logger)

        # 读取模型点云坐标并转为 numpy 数组。
        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        # 读取场景点云坐标并转为 numpy 数组。
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)
        # 计算当前样本的评估指标。
        m = compute_metrics(model_pts, scene_pts, T_pred, T_gt=T_gt, inlier_radius=5.0)

        # 将当前样本结果整理为字典。
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
        # 将当前结果加入总结果列表。
        results.append(rec)

    # 构造批处理结果 JSON 输出路径。
    out_json = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_batch.json")
    # 保存所有批处理结果及 GT 统计信息。
    save_json(out_json, {"results": results, "gt_ok": n_gt_ok, "gt_fail": n_gt_fail, "use_bop_gt": use_bop_gt, "t_scale": float(args.t_scale)})
    # 记录批处理结果保存路径。
    logger.info(f"Saved batch results to {out_json}")
    # 如果使用了 BOP GT，则在日志中补充成功与失败统计。
    if use_bop_gt:
        logger.info(f"BOP GT: ok={n_gt_ok} fail={n_gt_fail} (t_scale={args.t_scale})")


# 当脚本被直接运行时，调用主函数。
if __name__ == "__main__":
    # 启动批处理流程。
    main()
