# 导入 os 模块，用于路径拼接和文件存在性判断。
import os
# 导入 sys 模块，用于修改模块搜索路径。
import sys
# 导入 pandas 并命名为 pd，用于读取表格数据。
import pandas as pd
# 导入 numpy 并命名为 np，用于生成坐标数组。
import numpy as np
# 导入 matplotlib.pyplot，用于绘图。
import matplotlib.pyplot as plt

# 获取项目根目录。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录加入 Python 模块搜索路径。
sys.path.append(ROOT)

# 导入目录创建工具函数。
from ppf.utils import ensure_dir


# 定义绘制带误差棒柱状图的函数。
def barplot(metric_mean_col: str, metric_std_col: str, title: str, ylabel: str, out_name: str, df: pd.DataFrame):
    # 读取方法名称列表。
    methods = df["Method"].tolist()
    # 读取指定均值列，并转为浮点数组。
    means = df[metric_mean_col].astype(float).values
    # 读取指定标准差列，并转为浮点数组。
    stds = df[metric_std_col].astype(float).values

    # 生成横坐标位置数组。
    x = np.arange(len(methods))
    # 创建一个新的图形窗口。
    plt.figure()
    # 绘制柱状图，并给每个柱子添加误差棒。
    plt.bar(x, means, yerr=stds, capsize=4)
    # 设置横坐标刻度和标签显示方式。
    plt.xticks(x, methods, rotation=25, ha="right")
    # 设置图标题。
    plt.title(title)
    # 设置纵轴标签。
    plt.ylabel(ylabel)
    # 自动优化布局，避免标签重叠。
    plt.tight_layout()

    # 设置图像输出目录。
    out_dir = os.path.join(ROOT, "experiments", "figures")
    # 确保图像输出目录存在。
    ensure_dir(out_dir)
    # 拼接 PNG 输出路径。
    png_path = os.path.join(out_dir, out_name + ".png")
    # 拼接 PDF 输出路径。
    pdf_path = os.path.join(out_dir, out_name + ".pdf")
    # 保存 PNG 图片。
    plt.savefig(png_path, dpi=200)
    # 保存 PDF 图片。
    plt.savefig(pdf_path)
    # 关闭当前图形，释放资源。
    plt.close()
    # 打印 PNG 文件保存路径。
    print("Saved:", png_path)
    # 打印 PDF 文件保存路径。
    print("Saved:", pdf_path)


# 定义主函数，负责读取统计结果并绘制图表。
def main():
    # 构造聚合结果 CSV 路径。
    csv_path = os.path.join(ROOT, "experiments", "tables", "ablation_results.csv")
    # 如果 CSV 不存在，则提示应先生成表格。
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing table csv: {csv_path}. Run scripts/generate_tables.py first.")
    # 读取 CSV 为 DataFrame。
    df = pd.read_csv(csv_path)

    # 绘制 ADD 指标柱状图。
    barplot("ADD_mean", "ADD_std", "ADD Comparison", "ADD (higher is better)", "ablation_add", df)
    # 绘制旋转误差柱状图。
    barplot("RotErr_mean", "RotErr_std", "Rotation Error Comparison", "Rotation Error (deg, lower is better)", "ablation_roterr", df)
    # 绘制运行时间柱状图。
    barplot("Time_mean", "Time_std", "Runtime Comparison", "Time (s, lower is better)", "ablation_time", df)


# 当脚本被直接运行时，调用主函数。
if __name__ == "__main__":
    # 启动绘图流程。
    main()
