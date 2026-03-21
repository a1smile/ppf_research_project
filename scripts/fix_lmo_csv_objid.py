# 导入 os 模块，用于路径拼接和目录处理。
import os
# 导入 sys 模块，用于修改模块搜索路径。
import sys
# 导入 argparse 模块，用于解析命令行参数。
import argparse
# 导入 pandas 并命名为 pd，用于读取和处理 CSV 数据。
import pandas as pd

# 获取当前脚本上两级目录，即项目根目录。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录加入 Python 模块搜索路径。
sys.path.append(ROOT)

# 导入 BOP 真值相关辅助函数。
from ppf.bop_gt import scene_dir_from_depth_path, try_get_bop_gt_pose
# 导入目录创建工具函数。
from ppf.utils import ensure_dir


# 定义主函数，封装脚本执行流程。
def main():
    # 创建命令行参数解析器。
    ap = argparse.ArgumentParser()
    # 添加输入 CSV 路径参数，必须提供。
    ap.add_argument("--in_csv", type=str, required=True)
    # 添加输出 CSV 路径参数，必须提供。
    ap.add_argument("--out_csv", type=str, required=True)
    # 添加平移缩放参数，用于处理毫米或米单位。
    ap.add_argument("--t_scale", type=float, default=1.0, help="1.0 for mm, 0.001 for meters")
    # 解析命令行参数。
    args = ap.parse_args()

    # 读取输入 CSV 文件。
    df = pd.read_csv(args.in_csv)

    # 检查获取 BOP 真值所需列是否存在。
    for c in ["depth_path", "frame_id", "obj_token"]:
        # 若缺少必要列则报错。
        if c not in df.columns:
            raise ValueError(f"Input CSV missing required column for BOP GT: {c}")

    # 保存解析出的 BOP obj_id。
    obj_ids = []
    # 保存每条记录是否成功读取 GT。
    gt_ok = []
    # 保存每条记录的 GT 错误信息。
    gt_err = []
    # 保存每条记录展开后的 GT 位姿矩阵。
    gt_T = []

    # 逐行遍历输入表格。
    for i, r in df.iterrows():
        # 读取深度图路径。
        depth_path = str(r["depth_path"])
        # 读取帧编号。
        frame_id = int(r["frame_id"])
        # 读取目标标记编号。
        gt_id = int(r["obj_token"])

        # 根据深度图路径推断场景目录。
        scene_dir = scene_dir_from_depth_path(depth_path)
        # 尝试从 BOP 标注中获取 obj_id 和 GT 位姿。
        obj_id, T_gt, err = try_get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=float(args.t_scale))

        # 如果成功获取到 obj_id 和 GT 位姿，则记录成功结果。
        if err is None and obj_id is not None and T_gt is not None:
            # 保存修正后的 obj_id。
            obj_ids.append(int(obj_id))
            # 标记 GT 获取成功。
            gt_ok.append(True)
            # 成功时错误信息为空字符串。
            gt_err.append("")
            # 将 4x4 位姿矩阵拉平成一维列表后保存。
            gt_T.append(T_gt.reshape(-1).tolist())
        else:
            # 若 GT 获取失败，则优先保留原 CSV 中的 obj_id，否则填入 -1。
            obj_ids.append(int(r["obj_id"]) if "obj_id" in df.columns else -1)
            # 标记 GT 获取失败。
            gt_ok.append(False)
            # 保存失败原因。
            gt_err.append(str(err))
            # 失败时不保存 GT 位姿。
            gt_T.append(None)

    # 复制原始 DataFrame，避免原地修改。
    df_out = df.copy()
    # 写入修正后的 BOP obj_id 列。
    df_out["obj_id_bop"] = obj_ids
    # 写入 GT 成功标记列。
    df_out["gt_ok"] = gt_ok
    # 写入 GT 错误信息列。
    df_out["gt_err"] = gt_err
    # 写入展开后的 GT 位姿列。
    df_out["gt_T_flat"] = gt_T

    # 确保输出目录存在。
    ensure_dir(os.path.dirname(args.out_csv))
    # 将结果保存为 UTF-8 编码的 CSV。
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    # 打印保存路径。
    print("Saved fixed CSV:", args.out_csv)
    # 打印 GT 成功和失败数量统计。
    print("GT ok:", sum(gt_ok), "GT fail:", len(gt_ok) - sum(gt_ok))


# 当脚本被直接运行时，调用主函数。
if __name__ == "__main__":
    # 启动主流程。
    main()
