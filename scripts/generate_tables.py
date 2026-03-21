# 导入 os 模块，用于路径处理。
import os
# 导入 sys 模块，用于修改模块搜索路径。
import sys
# 导入 glob 模块，用于批量匹配结果文件。
import glob
# 导入 json 模块，用于读取 JSON 结果。
import json
# 导入 pandas 并命名为 pd，用于表格数据处理。
import pandas as pd
# 导入 numpy 并命名为 np，用于数值计算和 NaN 判断。
import numpy as np

# 获取项目根目录。
ROOT = os.path.dirname(os.path.dirname(__file__))
# 将项目根目录加入 Python 模块搜索路径。
sys.path.append(ROOT)

# 导入目录创建工具函数。
from ppf.utils import ensure_dir


# 定义方法的固定显示顺序。
ORDER = ["baseline", "plus_rsmrq", "plus_robustvote", "plus_kde", "full"]


# 计算一个数值序列的均值和标准差。
def mean_std(series: pd.Series):
    # 将输入序列尽量转换为数值，无法转换的项记为 NaN。
    s = pd.to_numeric(series, errors="coerce")
    # 去掉 NaN 项，只保留有效数值。
    s = s[~np.isnan(s.values)]
    # 如果没有有效值，则返回 NaN。
    if len(s) == 0:
        return float("nan"), float("nan")
    # 返回均值和样本标准差；若只有一个样本则标准差设为 0。
    return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0


# 将均值和标准差格式化为普通文本字符串。
def format_pm_plain(mean: float, std: float, nd: int = 3) -> str:
    # 如果均值不是有效数字，则直接返回 NaN。
    if np.isnan(mean):
        return "NaN"
    # 按给定位数格式化输出。
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


# 将均值和标准差格式化为适用于 LaTeX 的字符串。
def format_pm_tex(mean: float, std: float, nd: int = 3) -> str:
    # 如果均值不是有效数字，则直接返回 NaN。
    if np.isnan(mean):
        return "NaN"
    # 使用 LaTeX 的 \pm 记号格式化输出。
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


# 定义主函数，负责从 JSON 结果生成多种表格文件。
def main():
    # 设置实验结果目录。
    results_dir = os.path.join(ROOT, "experiments", "results")
    # 设置表格输出目录。
    tables_dir = os.path.join(ROOT, "experiments", "tables")
    # 确保表格输出目录存在。
    ensure_dir(tables_dir)

    # 获取结果目录下所有 JSON 文件，并按名称排序。
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    # 若没有结果文件，则报错提示。
    if not files:
        raise FileNotFoundError(f"No result json found in {results_dir}")

    # 用于收集所有读取到的实验记录。
    recs = []
    # 逐个读取 JSON 文件。
    for fp in files:
        # 以 UTF-8 编码打开结果文件。
        with open(fp, "r", encoding="utf-8") as f:
            # 将读取到的 JSON 对象加入记录列表。
            recs.append(json.load(f))

    # 将记录列表转换为 DataFrame。
    df = pd.DataFrame(recs)

    # 用于收集按方法聚合后的结果。
    rows = []
    # 按 method 分组统计各项指标。
    for method, g in df.groupby("method"):
        # 计算 ADD 指标的均值和标准差。
        add_m, add_s = mean_std(g["ADD"])
        # 计算旋转误差的均值和标准差。
        r_m, r_s = mean_std(g["rotation_error_deg"])
        # 计算平移误差的均值和标准差。
        t_m, t_s = mean_std(g["translation_error"])
        # 计算总耗时的均值和标准差。
        time_m, time_s = mean_std(g["total_time"])

        # 将当前方法的聚合结果整理为一行字典。
        rows.append({
            "Method": method,
            "ADD_mean": add_m, "ADD_std": add_s,
            "RotErr_mean": r_m, "RotErr_std": r_s,
            "TransErr_mean": t_m, "TransErr_std": t_s,
            "Time_mean": time_m, "Time_std": time_s
        })

    # 将聚合结果转换为输出 DataFrame。
    out = pd.DataFrame(rows)

    # 定义自定义排序函数，使已知方法按预设顺序排列。
    def sort_key(m):
        # 如果方法名在预设顺序中，则返回其顺序索引。
        if m in ORDER:
            return ORDER.index(m)
        # 未知方法统一排在最后。
        return 999

    # 按 Method 列使用自定义排序，并重置索引。
    out = out.sort_values(by="Method", key=lambda s: s.apply(sort_key)).reset_index(drop=True)

    # 构造输出 CSV 路径。
    csv_path = os.path.join(tables_dir, "ablation_results.csv")
    # 保存聚合结果为 CSV 文件。
    out.to_csv(csv_path, index=False, encoding="utf-8")
    # 打印 CSV 保存路径。
    print("Saved:", csv_path)

    # 构造 Markdown 表格输出路径。
    md_path = os.path.join(tables_dir, "ablation_results.md")
    # 打开 Markdown 文件进行写入。
    with open(md_path, "w", encoding="utf-8") as f:
        # 写入 Markdown 表头。
        f.write("| Method | ADD↑ | RotErr↓ | TransErr↓ | Time(s)↓ |\n")
        # 写入对齐控制行。
        f.write("|---|---:|---:|---:|---:|\n")
        # 逐行写入每个方法的结果。
        for _, r in out.iterrows():
            # 将每个方法的均值和标准差按表格格式写入文件。
            f.write(
                f"| {r['Method']} | {format_pm_plain(r['ADD_mean'], r['ADD_std'],2)} | "
                f"{format_pm_plain(r['RotErr_mean'], r['RotErr_std'],2)} | "
                f"{format_pm_plain(r['TransErr_mean'], r['TransErr_std'],3)} | "
                f"{format_pm_plain(r['Time_mean'], r['Time_std'],3)} |\n"
            )
    # 打印 Markdown 文件保存路径。
    print("Saved:", md_path)

    # 找到 ADD 的最佳值，数值越大越好。
    best_add = out["ADD_mean"].max(skipna=True)
    # 找到旋转误差的最佳值，数值越小越好。
    best_r = out["RotErr_mean"].min(skipna=True)
    # 找到平移误差的最佳值，数值越小越好。
    best_t = out["TransErr_mean"].min(skipna=True)
    # 找到总耗时的最佳值，数值越小越好。
    best_time = out["Time_mean"].min(skipna=True)

    # 判断当前值是否应在表格中高亮为最佳结果。
    def maybe_bold(val, best, higher=True):
        # 如果当前值或最佳值为 NaN，则不加粗。
        if np.isnan(val) or np.isnan(best):
            return False
        # 根据指标方向判断是否达到最佳值。
        return (val >= best - 1e-12) if higher else (val <= best + 1e-12)

    # 构造 LaTeX 表格输出路径。
    tex_path = os.path.join(tables_dir, "ablation_results.tex")
    # 打开 LaTeX 文件进行写入。
    with open(tex_path, "w", encoding="utf-8") as f:
        # 写入表格环境开头。
        f.write("\\begin{table}[t]\n\\centering\n")
        # 写入列格式和顶部分隔线。
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        # 写入表头行。
        f.write("Method & ADD$\\uparrow$ & RotErr$\\downarrow$ & TransErr$\\downarrow$ & Time(s)$\\downarrow$ \\\\\n\\midrule\n")
        # 逐行写入每个方法的 LaTeX 表格内容。
        for _, r in out.iterrows():
            # 格式化 ADD 字段。
            add_str = format_pm_tex(r["ADD_mean"], r["ADD_std"], 2)
            # 格式化旋转误差字段。
            rot_str = format_pm_tex(r["RotErr_mean"], r["RotErr_std"], 2)
            # 格式化平移误差字段。
            tr_str  = format_pm_tex(r["TransErr_mean"], r["TransErr_std"], 3)
            # 格式化时间字段。
            ti_str  = format_pm_tex(r["Time_mean"], r["Time_std"], 3)

            # 若当前 ADD 为最优，则加粗显示。
            if maybe_bold(r["ADD_mean"], best_add, higher=True):
                add_str = "\\textbf{" + add_str + "}"
            # 若当前旋转误差为最优，则加粗显示。
            if maybe_bold(r["RotErr_mean"], best_r, higher=False):
                rot_str = "\\textbf{" + rot_str + "}"
            # 若当前平移误差为最优，则加粗显示。
            if maybe_bold(r["TransErr_mean"], best_t, higher=False):
                tr_str = "\\textbf{" + tr_str + "}"
            # 若当前时间为最优，则加粗显示。
            if maybe_bold(r["Time_mean"], best_time, higher=False):
                ti_str = "\\textbf{" + ti_str + "}"

            # 写入当前方法对应的一行 LaTeX 内容。
            f.write(f"{r['Method']} & {add_str} & {rot_str} & {tr_str} & {ti_str} \\\\\n")

        # 写入表格结束标记、标题和标签。
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Ablation Study on Proposed PPF Enhancements.}\n")
        f.write("\\label{tab:ablation}\n\\end{table}\n")
    # 打印 LaTeX 文件保存路径。
    print("Saved:", tex_path)


# 当脚本被直接运行时，调用主函数。
if __name__ == "__main__":
    # 启动表格生成流程。
    main()
