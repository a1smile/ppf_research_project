# 导入 os 模块，用于处理路径等操作系统相关功能。
import os
# 导入 sys 模块，用于访问解释器参数和模块搜索路径。
import sys
# 导入 argparse 模块，用于解析命令行参数。
import argparse
# 导入 logging 模块，用于定义日志等级等日志相关配置。
import logging
# 导入 numpy 库并命名为 np，通常用于数值计算。
import numpy as np
# 导入 open3d 库并命名为 o3d，用于点云读取与可视化。
import open3d as o3d

# 说明下面的代码用于让当前脚本能够导入项目根目录下的模块。
# 获取当前文件所在目录的上一级目录的上一级目录，即项目根目录。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录追加到 Python 模块搜索路径中，便于导入项目内部包。
sys.path.append(ROOT)

# 从 ppf.io 模块中导入 load_config 函数，用于加载配置文件。
from ppf.io import load_config
# 从 ppf.utils 模块中导入 setup_logger 和 ensure_dir，分别用于初始化日志和确保目录存在。
from ppf.utils import setup_logger, ensure_dir
# 从 ppf.registration 模块中导入 run_registration 函数，用于执行配准主流程。
from ppf.registration import run_registration


# 定义主函数，封装脚本的主要执行逻辑。
def main():
    # 创建命令行参数解析器对象。
    ap = argparse.ArgumentParser()
    # 添加 --config 参数，类型为字符串，默认配置文件为 configs/default.yaml。
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    # 添加 --model 参数，类型为字符串，且为必填，用于指定模型点云路径。
    ap.add_argument("--model", type=str, required=True)
    # 添加 --scene 参数，类型为字符串，且为必填，用于指定场景点云路径。
    ap.add_argument("--scene", type=str, required=True)
    # 添加 --log_name 参数，类型为字符串，默认日志名为 demo。
    ap.add_argument("--log_name", type=str, default="demo")
    # 解析命令行参数，并将结果保存到 args 中。
    args = ap.parse_args()

    # 加载配置文件；如果给的是相对路径，则拼接到项目根目录下，否则直接使用绝对路径。
    cfg = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)

    # 从配置中读取日志输出目录。
    out_logs = cfg["output"]["logs_dir"]
    # 确保日志输出目录存在，不存在时自动创建。
    ensure_dir(out_logs)
    # 拼接最终日志文件路径，文件名由 log_name 参数决定。
    log_path = os.path.join(out_logs, f"{args.log_name}.log")
    # 按 INFO 级别创建并返回日志记录器。
    logger = setup_logger(log_path, level=logging.INFO)

    # 记录演示流程开始的标识信息。
    logger.info("=== RUN DEMO ===")
    # 记录当前使用的配置文件路径。
    logger.info(f"config={args.config}")
    # 记录当前使用的模型点云路径。
    logger.info(f"model={args.model}")
    # 记录当前使用的场景点云路径。
    logger.info(f"scene={args.scene}")
    # 记录配置中若干可选功能开关的启用状态，不存在时默认记为 False。
    logger.info(f"enable_rsmrq={cfg.get('enable_rsmrq', False)} enable_robust_vote={cfg.get('enable_robust_vote', False)} enable_kde_refine={cfg.get('enable_kde_refine', False)}")
    # 记录完整配置内容，便于排查问题和复现实验。
    logger.info(f"full cfg:\n{cfg}")

    # 调用配准主函数，输入模型、场景和配置，并传入日志器；返回预测变换、输出模型、调试信息和统计信息。
    T_pred, out_model, debug, stats = run_registration(args.model, args.scene, cfg, logger=logger)

    # 记录预测得到的位姿变换矩阵。
    logger.info(f"T_pred=\n{T_pred}")
    # 记录统计信息，例如耗时或候选数量等。
    logger.info(f"stats={stats}")
    # 记录调试信息，便于分析内部流程结果。
    logger.info(f"debug={debug}")

    # 如果配置中 visualize 为真，则执行点云可视化。
    if bool(cfg.get("visualize", False)):
        # 读取场景点云文件。
        scene = o3d.io.read_point_cloud(args.scene)
        # 将配准输出的模型数据构造成 Open3D 点云对象。
        model_vis = o3d.geometry.PointCloud(out_model)
        # 将模型点云统一涂成红色，便于和场景区分。
        model_vis.paint_uniform_color([1.0, 0.2, 0.2])
        # 同时显示场景点云和模型点云。
        o3d.visualization.draw_geometries([scene, model_vis])


# 当该文件作为主程序直接运行时，执行 main 函数。
if __name__ == "__main__":
    # 调用主函数，启动整个演示流程。
    main()
