import json
import os
import numpy as np
import matplotlib.pyplot as plt

files = {
    "Baseline": "stanford_baseline_batch.json",
    "w/o Pose": "stanford_no_pose_pipeline_fixed_batch.json",
    "w/o Robust Vote": "stanford_no_robust_vote_batch.json",
    "w/o RSMRQ": "stanford_no_rsmrq_batch.json",
    "Ours": "stanford_retrieval_cached_batch.json",
}

OUT_PDF = "boxplot_errors_final.pdf"
OUT_SVG = "boxplot_errors_final.svg"

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

plt.rcParams.update(STYLE)


def load_metric(path, key):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported JSON format")

    values = []
    for x in records:
        if not isinstance(x, dict):
            continue

        if "metrics" in x:
            val = x["metrics"].get(key, None)
        else:
            val = x.get(key.lower(), None)

        if val is not None:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                pass

    return np.array(values)


# Load data
add_data = []
rot_data = []
labels = []

for name, path in files.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}")
        continue

    add_vals = load_metric(path, "ADD")
    rot_vals = load_metric(path, "rotation_error_deg")

    add_data.append(add_vals)
    rot_data.append(rot_vals)
    labels.append(name)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))


def draw_boxplot(ax, data, title, ylabel):
    positions = np.arange(1, len(data) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
    )

    ours_idx = labels.index("Ours") if "Ours" in labels else None

    for i, box in enumerate(bp["boxes"]):
        if i == ours_idx:
            box.set_hatch("///")
            box.set_linewidth(1.5)
        else:
            box.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_linewidth(1.5)

    ax.set_title(title, pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ticklabels = ax.set_xticklabels(labels, rotation=25, ha="right")

    if ours_idx is not None:
        ticklabels[ours_idx].set_fontweight("bold")

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


draw_boxplot(axes[0], add_data, "(a) ADD Distribution", "ADD Error")
draw_boxplot(axes[1], rot_data, "(b) Rotation Error Distribution", "Degrees")

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
