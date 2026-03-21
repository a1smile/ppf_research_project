# 导入 os 模块，用于路径处理。
import os
# 导入 sys 模块，用于修改模块搜索路径。
import sys
# 导入 argparse 模块，用于解析命令行参数。
import argparse
# 导入 logging 模块，用于定义日志等级。
import logging
# 导入 glob 模块，用于匹配多个配置文件。
import glob
# 导入 math 模块，供脚本中的数学逻辑使用。
import math
# 导入 numpy 并命名为 np，用于数值与数组处理。
import numpy as np
# 导入 open3d 并命名为 o3d，用于读取点云文件。
import open3d as o3d

# 获取项目根目录路径。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录加入 Python 模块搜索路径。
sys.path.append(ROOT)

# 导入配置加载函数。
from ppf.io import load_config
# 导入日志、目录创建和 JSON 保存相关工具函数。
from ppf.utils import setup_logger, ensure_dir, save_json
# 导入配准主流程函数。
from ppf.registration import run_registration
# 导入评估指标计算函数。
from ppf.metrics import compute_metrics


# 根据基础配置生成若干消融实验配置变体。
def method_variants(base_cfg: dict):
    """
    Must run:
    - Baseline (all off)
    - +RS-MRQ
    - +RobustVote
    - +KDE
    - Full (all on)
    """
    # 定义内部函数，用于深拷贝基础配置。
    def clone():
        # 在局部导入 copy 模块。
        import copy
        # 返回基础配置的深拷贝副本。
        return copy.deepcopy(base_cfg)

    # 用于保存所有方法变体。
    variants = []

    # 构造基线配置，即关闭所有增强模块。
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = False
    variants.append(("baseline", c))

    # 构造仅启用 RS-MRQ 的配置。
    c = clone()
    c["enable_rsmrq"] = True
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = False
    variants.append(("plus_rsmrq", c))

    # 构造仅启用 RobustVote 的配置。
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = True
    c["enable_kde_refine"] = False
    variants.append(("plus_robustvote", c))

    # 构造仅启用 KDE refine 的配置。
    c = clone()
    c["enable_rsmrq"] = False
    c["enable_robust_vote"] = False
    c["enable_kde_refine"] = True
    variants.append(("plus_kde", c))

    # 构造所有增强模块全部开启的完整配置。
    c = clone()
    c["enable_rsmrq"] = True
    c["enable_robust_vote"] = True
    c["enable_kde_refine"] = True
    variants.append(("full", c))

    # 返回所有方法及其配置。
    return variants


# 读取可选的 GT 位姿文件。
def load_optional_gt(gt_path: str):
    # 如果没有提供 GT 路径，则返回 None。
    if not gt_path:
        return None
    # 在函数内部导入 json 模块。
    import json
    # 打开 GT 位姿文件。
    with open(gt_path, "r", encoding="utf-8") as f:
        # 读取 JSON 数据。
        data = json.load(f)
    # 从字段 T 中读取 4x4 变换矩阵。
    T = np.array(data["T"], dtype=np.float64)
    # 若矩阵尺寸不是 4x4，则报错。
    if T.shape != (4, 4):
        raise ValueError("GT pose must be 4x4 in json field 'T'")
    # 返回 GT 位姿矩阵。
    return T


# 定义主函数，负责执行消融实验。
def main():
    # 创建命令行参数解析器。
    ap = argparse.ArgumentParser()
    # 添加配置文件路径参数。
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    # 添加模型点云路径参数，必须提供。
    ap.add_argument("--model", type=str, required=True)
    # 添加场景点云路径参数，必须提供。
    ap.add_argument("--scene", type=str, required=True)
    # 添加重复运行次数参数。
    ap.add_argument("--repeat", type=int, default=10)
    # 添加可选 GT 位姿 JSON 参数。
    ap.add_argument("--gt_pose", type=str, default="", help="optional json with {'T':[[...4x4...]]}")
    # 添加扫描额外消融配置的开关参数。
    ap.add_argument("--scan_ablation_configs", action="store_true", help="also scan configs/ablation_*.yaml and run them")
    # 解析命令行参数。
    args = ap.parse_args()

    # 根据输入路径得到配置文件绝对路径。
    cfg_path = os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config
    # 加载基础配置。
    base_cfg = load_config(cfg_path)

    # 读取结果输出目录。
    out_res = base_cfg["output"]["results_dir"]
    # 读取日志输出目录。
    out_logs = base_cfg["output"]["logs_dir"]
    # 确保结果输出目录存在。
    ensure_dir(out_res)
    # 确保日志输出目录存在。
    ensure_dir(out_logs)

    # 加载可选 GT 位姿。
    T_gt = load_optional_gt(args.gt_pose)

    # 用于收集所有待运行的方法配置。
    methods = []
    # 先加入固定要求的 5 种消融方法。
    methods.extend(method_variants(base_cfg))

    # 如果开启扫描开关，则额外读取 configs 下的 ablation_*.yaml。
    if args.scan_ablation_configs:
        # 遍历所有匹配到的消融配置文件。
        for y in sorted(glob.glob(os.path.join(ROOT, "configs", "ablation_*.yaml"))):
            # 取文件名作为方法名。
            name = os.path.splitext(os.path.basename(y))[0]
            # 加载该配置文件。
            c = load_config(y)
            # 将方法名和配置加入列表。
            methods.append((name, c))

    # 依次运行每种方法，并进行多次重复实验。
    for method_name, cfg in methods:
        # 对当前方法执行 repeat 次实验。
        for run_id in range(args.repeat):
            # 根据配置中的基础种子和 run_id 生成本次运行种子。
            seed = int(cfg.get("seed", 0)) + run_id
            # 将本次种子写回配置。
            cfg["seed"] = seed

            # 构造当前实验的日志文件路径。
            log_path = os.path.join(out_logs, f"{method_name}_run{run_id}.log")
            # 创建日志记录器。
            logger = setup_logger(log_path, level=logging.INFO)

            # 记录消融实验开始。
            logger.info("=== RUN ABLATION ===")
            # 记录方法名、运行编号和随机种子。
            logger.info(f"method={method_name} run_id={run_id} seed={seed}")
            # 记录模型路径。
            logger.info(f"model={args.model}")
            # 记录场景路径。
            logger.info(f"scene={args.scene}")
            # 记录关键模块开关状态。
            logger.info(f"enable_rsmrq={cfg.get('enable_rsmrq', False)} enable_robust_vote={cfg.get('enable_robust_vote', False)} enable_kde_refine={cfg.get('enable_kde_refine', False)}")
            # 记录完整配置内容。
            logger.info(f"cfg:\n{cfg}")

            # 执行配准流程，返回预测位姿、输出模型、调试信息和统计信息。
            T_pred, out_model, debug, stats = run_registration(args.model, args.scene, cfg, logger=logger)

            # 读取模型点云并转为 numpy 数组。
            model_pts = np.asarray(o3d.io.read_point_cloud(args.model).points, dtype=np.float64)
            # 读取场景点云并转为 numpy 数组。
            scene_pts = np.asarray(o3d.io.read_point_cloud(args.scene).points, dtype=np.float64)

            # 计算当前实验的评估指标。
            metrics = compute_metrics(model_pts, scene_pts, T_pred, T_gt=T_gt, inlier_radius=5.0)

            # 将当前实验结果整理为字典。
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

            # 构造当前实验结果 JSON 的输出路径。
            out_json = os.path.join(out_res, f"{method_name}_run{run_id}.json")
            # 保存当前实验结果。
            save_json(out_json, rec)
            # 在日志中记录保存路径。
            logger.info(f"Saved result: {out_json}")

            # 按要求记录关键统计信息。
            logger.info(f"[RS-MRQ] candidate_inflation_mean={stats.candidate_inflation_mean:.3f}")
            # 如果启用了 RobustVote，则记录对应摘要。
            if cfg.get("enable_robust_vote", False):
                logger.info(f"[RobustVote] summary={stats.robust_vote_summary}")
            # 如果启用了 KDE refine，则记录调用次数。
            if cfg.get("enable_kde_refine", False):
                logger.info(f"[KDERefine] calls={stats.kde_refine_calls}")
            # 记录建模、配准和总耗时。
            logger.info(f"[Timing] model_build={stats.model_build_time:.6f} reg={stats.registration_time:.6f} total={stats.total_time:.6f}")


# 当脚本被直接运行时，调用主函数。
if __name__ == "__main__":
    # 启动消融实验流程。
    main()
