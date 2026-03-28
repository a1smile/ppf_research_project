import csv
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "summary_stats.csv"
OUT_PDF = "tradeoff_scatter_final.pdf"
OUT_SVG = "tradeoff_scatter_final.svg"

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

methods_raw = [r["method"] for r in rows]
methods = [name_map.get(m, m) for m in methods_raw]

time_mean = np.array([float(r["time_mean"]) for r in rows], dtype=float)
add_mean = np.array([float(r["add_mean"]) for r in rows], dtype=float)
add_success = np.array([float(r["add_success"]) for r in rows], dtype=float)

ours_idx = methods_raw.index("ours") if "ours" in methods_raw else None

plt.rcParams.update(STYLE)


def padded_log_limits(values, pad_ratio=0.20, min_pad_decades=0.10):
    log_vals = np.log10(values)
    lo = float(np.min(log_vals))
    hi = float(np.max(log_vals))
    span = hi - lo
    pad = max(span * pad_ratio, min_pad_decades)
    return 10 ** (lo - pad), 10 ** (hi + pad)


def padded_linear_limits(values, pad_ratio=0.12, floor=None, ceiling=None):
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    ref = max(abs(lo), abs(hi), 1.0)
    pad = max(span * pad_ratio, 0.03 * ref)
    lo_pad = lo - pad
    hi_pad = hi + pad

    if floor is not None:
        lo_pad = max(floor, lo_pad)
    if ceiling is not None:
        hi_pad = min(ceiling, hi_pad)

    return lo_pad, hi_pad


def annotation_style(x, y, xs, ys, bold=False):
    log_xs = np.log10(xs)
    x_pos = (np.log10(x) - log_xs.min()) / (log_xs.max() - log_xs.min() + 1e-12)
    y_pos = (y - ys.min()) / (ys.max() - ys.min() + 1e-12)

    dx = 7 if x_pos < 0.80 else -7
    dy = 4 if y_pos < 0.82 else -5

    return {
        "xytext": (dx, dy),
        "textcoords": "offset points",
        "fontsize": 8.5 if bold else 8,
        "fontweight": "bold" if bold else "normal",
        "ha": "left" if dx > 0 else "right",
        "va": "bottom" if dy >= 0 else "top",
        "clip_on": False,
    }


def style_axes(ax):
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_tradeoff(y_values, y_label, title, out_pdf, out_svg, y_floor=None, y_ceiling=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.set_xscale("log")
    ax.set_xlim(*padded_log_limits(time_mean))
    ax.set_ylim(*padded_linear_limits(y_values, floor=y_floor, ceiling=y_ceiling))

    for i, name in enumerate(methods):
        is_ours = i == ours_idx
        ax.scatter(
            time_mean[i],
            y_values[i],
            s=95 if is_ours else 70,
            marker="o",
            zorder=4 if is_ours else 3,
        )
        ax.annotate(
            name,
            xy=(time_mean[i], y_values[i]),
            **annotation_style(time_mean[i], y_values[i], time_mean, y_values, bold=is_ours),
        )

    ax.set_xlabel("Registration Time (s, log scale)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.show()


plot_tradeoff(
    add_mean,
    "ADD Mean",
    "Accuracy-Efficiency Trade-off",
    OUT_PDF,
    OUT_SVG,
    y_floor=0.0,
)

plot_tradeoff(
    add_success,
    "Success Rate @ ADD<0.02",
    "Success-Efficiency Trade-off",
    "tradeoff_success_final.pdf",
    "tradeoff_success_final.svg",
    y_floor=0.0,
    y_ceiling=1.02,
)

print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
print("Saved: tradeoff_success_final.pdf")
print("Saved: tradeoff_success_final.svg")
