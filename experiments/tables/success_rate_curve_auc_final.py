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

# Sweep thresholds
T_MIN = 0.0
T_MAX = 0.10
N_THRESH = 300
thresholds = np.linspace(T_MIN, T_MAX, N_THRESH)

OUT_PDF = "success_rate_curve_auc_final.pdf"
OUT_SVG = "success_rate_curve_auc_final.svg"

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


def load_add_values(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unsupported JSON format in {path}")

    add_vals = []
    for x in records:
        if not isinstance(x, dict):
            continue
        if "metrics" in x and isinstance(x["metrics"], dict):
            val = x["metrics"].get("ADD", None)
        else:
            val = x.get("add", None)
        if val is not None:
            try:
                add_vals.append(float(val))
            except (TypeError, ValueError):
                pass

    if not add_vals:
        raise ValueError(f"No valid ADD values found in {path}")

    return np.array(add_vals, dtype=float)


def compute_success_curve(add_values, thresholds):
    return np.array([np.mean(add_values < t) for t in thresholds], dtype=float)


def auc_of_curve(xs, ys):
    raw_auc = np.trapz(ys, xs)
    normalized_auc = raw_auc / (xs[-1] - xs[0])
    return float(normalized_auc)


fig, ax = plt.subplots(figsize=(6.6, 4.6))
curves = {}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        continue

    add_vals = load_add_values(path)
    success = compute_success_curve(add_vals, thresholds)
    auc = auc_of_curve(thresholds, success)
    curves[name] = {"success": success, "auc": auc}

plot_order = ["Baseline", "w/o Pose", "w/o Robust Vote", "w/o RSMRQ", "Ours"]

for name in plot_order:
    if name not in curves:
        continue

    success = curves[name]["success"]
    auc = curves[name]["auc"]
    label = f"{name} (AUC={auc:.3f})"

    if name == "Ours":
        ax.plot(thresholds, success, linewidth=2.6, label=label)
    else:
        ax.plot(
            thresholds,
            success,
            linewidth=1.8,
            linestyle="--",
            label=label,
            alpha=0.95,
        )

ref_t = 0.02
ref_idx = int(np.argmin(np.abs(thresholds - ref_t)))
if "Ours" in curves:
    y_ref = curves["Ours"]["success"][ref_idx]
    ax.scatter([thresholds[ref_idx]], [y_ref], s=28, zorder=5)
    ax.annotate(
        f"Ours @ 0.02 = {y_ref:.3f}",
        xy=(thresholds[ref_idx], y_ref),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel("ADD Threshold")
ax.set_ylabel("Success Rate")
ax.set_xlim(T_MIN, T_MAX)
ax.set_ylim(0.0, 1.02)
ax.set_xticks(np.linspace(0.0, 0.10, 6))
ax.set_yticks(np.linspace(0.0, 1.0, 6))
ax.grid(alpha=0.22, linewidth=0.6)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(
    frameon=False,
    loc="lower right",
    handlelength=2.4,
    borderaxespad=0.5,
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
print("\nAUC summary:")
for name in plot_order:
    if name in curves:
        print(f"{name:<18} {curves[name]['auc']:.4f}")
