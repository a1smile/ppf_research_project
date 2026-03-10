import os
import sys
import glob
import json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.utils import ensure_dir


ORDER = ["baseline", "plus_rsmrq", "plus_robustvote", "plus_kde", "full"]


def mean_std(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[~np.isnan(s.values)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0


def format_pm_plain(mean: float, std: float, nd: int = 3) -> str:
    if np.isnan(mean):
        return "NaN"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def format_pm_tex(mean: float, std: float, nd: int = 3) -> str:
    if np.isnan(mean):
        return "NaN"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def main():
    results_dir = os.path.join(ROOT, "experiments", "results")
    tables_dir = os.path.join(ROOT, "experiments", "tables")
    ensure_dir(tables_dir)

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No result json found in {results_dir}")

    recs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            recs.append(json.load(f))

    df = pd.DataFrame(recs)

    rows = []
    for method, g in df.groupby("method"):
        add_m, add_s = mean_std(g["ADD"])
        r_m, r_s = mean_std(g["rotation_error_deg"])
        t_m, t_s = mean_std(g["translation_error"])
        time_m, time_s = mean_std(g["total_time"])

        rows.append({
            "Method": method,
            "ADD_mean": add_m, "ADD_std": add_s,
            "RotErr_mean": r_m, "RotErr_std": r_s,
            "TransErr_mean": t_m, "TransErr_std": t_s,
            "Time_mean": time_m, "Time_std": time_s
        })

    out = pd.DataFrame(rows)

    def sort_key(m):
        if m in ORDER:
            return ORDER.index(m)
        return 999

    out = out.sort_values(by="Method", key=lambda s: s.apply(sort_key)).reset_index(drop=True)

    csv_path = os.path.join(tables_dir, "ablation_results.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved:", csv_path)

    md_path = os.path.join(tables_dir, "ablation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | ADD↑ | RotErr↓ | TransErr↓ | Time(s)↓ |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in out.iterrows():
            f.write(
                f"| {r['Method']} | {format_pm_plain(r['ADD_mean'], r['ADD_std'],2)} | "
                f"{format_pm_plain(r['RotErr_mean'], r['RotErr_std'],2)} | "
                f"{format_pm_plain(r['TransErr_mean'], r['TransErr_std'],3)} | "
                f"{format_pm_plain(r['Time_mean'], r['Time_std'],3)} |\n"
            )
    print("Saved:", md_path)

    best_add = out["ADD_mean"].max(skipna=True)
    best_r = out["RotErr_mean"].min(skipna=True)
    best_t = out["TransErr_mean"].min(skipna=True)
    best_time = out["Time_mean"].min(skipna=True)

    def maybe_bold(val, best, higher=True):
        if np.isnan(val) or np.isnan(best):
            return False
        return (val >= best - 1e-12) if higher else (val <= best + 1e-12)

    tex_path = os.path.join(tables_dir, "ablation_results.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & ADD$\\uparrow$ & RotErr$\\downarrow$ & TransErr$\\downarrow$ & Time(s)$\\downarrow$ \\\\\n\\midrule\n")
        for _, r in out.iterrows():
            add_str = format_pm_tex(r["ADD_mean"], r["ADD_std"], 2)
            rot_str = format_pm_tex(r["RotErr_mean"], r["RotErr_std"], 2)
            tr_str  = format_pm_tex(r["TransErr_mean"], r["TransErr_std"], 3)
            ti_str  = format_pm_tex(r["Time_mean"], r["Time_std"], 3)

            if maybe_bold(r["ADD_mean"], best_add, higher=True):
                add_str = "\\textbf{" + add_str + "}"
            if maybe_bold(r["RotErr_mean"], best_r, higher=False):
                rot_str = "\\textbf{" + rot_str + "}"
            if maybe_bold(r["TransErr_mean"], best_t, higher=False):
                tr_str = "\\textbf{" + tr_str + "}"
            if maybe_bold(r["Time_mean"], best_time, higher=False):
                ti_str = "\\textbf{" + ti_str + "}"

            f.write(f"{r['Method']} & {add_str} & {rot_str} & {tr_str} & {ti_str} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Ablation Study on Proposed PPF Enhancements.}\n")
        f.write("\\label{tab:ablation}\n\\end{table}\n")
    print("Saved:", tex_path)


if __name__ == "__main__":
    main()
