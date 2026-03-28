import csv
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "summary_stats.csv"
OUT_SVG = "final_cvpr_style_enhanced.svg"
OUT_PDF = "final_cvpr_style_enhanced.pdf"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

name_map = {
    "baseline": "Baseline",
    "no_pose": "w/o Pose",
    "no_robust": "w/o Robust Vote",
    "no_rsmrq": "w/o RSMRQ",
    "ours": "Ours",
}

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError(f"No rows found in {CSV_PATH}")

methods_raw = [r["method"] for r in rows]
methods = [name_map.get(m, m) for m in methods_raw]
ours_idx = methods_raw.index("ours") if "ours" in methods_raw else None

plt.rcParams.update(STYLE)


def get_vals(key):
    if key not in rows[0]:
        raise KeyError(f"Missing required column: {key}")
    return np.array([float(r[key]) for r in rows], dtype=float)


def get_first_available(keys):
    for key in keys:
        if key in rows[0]:
            return np.array([float(r[key]) for r in rows], dtype=float), key
    return None, None


def get_median_iqr_errors(prefix, median):
    q1, q1_key = get_first_available([
        f"{prefix}_q1",
        f"{prefix}_p25",
        f"{prefix}_25",
        f"{prefix}_quartile1",
    ])
    q3, q3_key = get_first_available([
        f"{prefix}_q3",
        f"{prefix}_p75",
        f"{prefix}_75",
        f"{prefix}_quartile3",
    ])

    if q1 is not None and q3 is not None:
        lower = np.clip(median - q1, a_min=0, a_max=None)
        upper = np.clip(q3 - median, a_min=0, a_max=None)
        return np.vstack([lower, upper]), f"{q1_key}/{q3_key}"

    iqr, iqr_key = get_first_available([
        f"{prefix}_iqr",
        f"{prefix}_IQR",
    ])
    if iqr is not None:
        half_iqr = np.clip(iqr / 2.0, a_min=0, a_max=None)
        print(
            f"[WARN] Using symmetric IQR/2 for {prefix} because Q1/Q3 columns are unavailable."
            f" Found: {iqr_key}"
        )
        return np.vstack([half_iqr, half_iqr]), iqr_key

    raise KeyError(
        "Missing quartile information for median error bars. "
        f"Please add {prefix}_q1/{prefix}_q3 (preferred) or {prefix}_iqr to {CSV_PATH}."
    )


def best_indices(values, higher_is_better=False, atol=1e-12):
    target = np.max(values) if higher_is_better else np.min(values)
    return [i for i, v in enumerate(values) if abs(v - target) <= atol]


fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.25))
axes = axes.ravel()

metrics = [
    {
        "panel": "(a)",
        "title": r"ADD $\downarrow$",
        "prefix": "add",
        "mean_key": "add_mean",
        "median_key": "add_median",
        "std_key": "add_std",
        "ylabel": "Error",
        "log": False,
        "higher_is_better": False,
    },
    {
        "panel": "(b)",
        "title": r"ADD-S $\downarrow$",
        "prefix": "add_s",
        "mean_key": "add_s_mean",
        "median_key": "add_s_median",
        "std_key": "add_s_std",
        "ylabel": "Error",
        "log": False,
        "higher_is_better": False,
    },
    {
        "panel": "(c)",
        "title": r"Rotation Error $\downarrow$",
        "prefix": "rot",
        "mean_key": "rot_mean",
        "median_key": "rot_median",
        "std_key": "rot_std",
        "ylabel": "Degrees",
        "log": False,
        "higher_is_better": False,
    },
    {
        "panel": "(d)",
        "title": r"Registration Time $\downarrow$",
        "prefix": "time",
        "mean_key": "time_mean",
        "median_key": "time_median",
        "std_key": "time_std",
        "ylabel": "Seconds (log)",
        "log": True,
        "higher_is_better": False,
    },
]

x = np.arange(len(methods))
w = 0.34

legend_handles = None
legend_labels = None

for ax, cfg in zip(axes, metrics):
    mean = get_vals(cfg["mean_key"])
    median = get_vals(cfg["median_key"])
    std = get_vals(cfg["std_key"])
    median_yerr, median_err_source = get_median_iqr_errors(cfg["prefix"], median)

    bars_mean = ax.bar(
        x - w / 2,
        mean,
        w,
        yerr=std,
        capsize=2,
        label="Mean ± STD",
    )
    bars_median = ax.bar(
        x + w / 2,
        median,
        w,
        yerr=median_yerr,
        capsize=2,
        alpha=0.8,
        label="Median ± IQR",
    )

    if ours_idx is not None:
        bars_mean[ours_idx].set_hatch("///")
        bars_median[ours_idx].set_hatch("///")

    mean_best = best_indices(mean, higher_is_better=cfg["higher_is_better"])
    median_best = best_indices(median, higher_is_better=cfg["higher_is_better"])

    def add_best_markers(bars, idxs, yvals, upper_err):
        for i in idxs:
            y = yvals[i] + upper_err[i]
            if cfg["log"]:
                y = max(y, yvals[i] * 1.05)
                ax.annotate(
                    "★",
                    (bars[i].get_x() + bars[i].get_width() / 2, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ax.annotate(
                    "★",
                    (bars[i].get_x() + bars[i].get_width() / 2, y),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_best_markers(bars_mean, mean_best, mean, std)
    add_best_markers(bars_median, median_best, median, median_yerr[1])

    if cfg["log"]:
        ax.set_yscale("log")

    ax.set_title(f"{cfg['panel']} {cfg['title']}", pad=3)
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xticks(x)
    tick_texts = ax.set_xticklabels(methods, rotation=24, ha="right")

    if ours_idx is not None:
        tick_texts[ours_idx].set_fontweight("bold")

    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    median_upper = median + median_yerr[1]
    ymax = max(np.max(mean + std), np.max(median_upper))
    if cfg["log"]:
        positives = np.concatenate([mean[mean > 0], median[median > 0]])
        ymin = max(1e-4, positives.min() * 0.6)
        ax.set_ylim(ymin, ymax * 3.0)
    else:
        ax.set_ylim(0, ymax * 1.22)

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    print(f"[INFO] {cfg['prefix']}: median error bars from {median_err_source}")

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.985),
    handlelength=1.6,
    columnspacing=1.5,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

print("Saved:", OUT_SVG)
print("Saved:", OUT_PDF)
