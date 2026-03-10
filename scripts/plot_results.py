import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from ppf.utils import ensure_dir


def barplot(metric_mean_col: str, metric_std_col: str, title: str, ylabel: str, out_name: str, df: pd.DataFrame):
    methods = df["Method"].tolist()
    means = df[metric_mean_col].astype(float).values
    stds = df[metric_std_col].astype(float).values

    x = np.arange(len(methods))
    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_dir = os.path.join(ROOT, "experiments", "figures")
    ensure_dir(out_dir)
    png_path = os.path.join(out_dir, out_name + ".png")
    pdf_path = os.path.join(out_dir, out_name + ".pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def main():
    csv_path = os.path.join(ROOT, "experiments", "tables", "ablation_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing table csv: {csv_path}. Run scripts/generate_tables.py first.")
    df = pd.read_csv(csv_path)

    barplot("ADD_mean", "ADD_std", "ADD Comparison", "ADD (higher is better)", "ablation_add", df)
    barplot("RotErr_mean", "RotErr_std", "Rotation Error Comparison", "Rotation Error (deg, lower is better)", "ablation_roterr", df)
    barplot("Time_mean", "Time_std", "Runtime Comparison", "Time (s, lower is better)", "ablation_time", df)


if __name__ == "__main__":
    main()
