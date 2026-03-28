import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

DEFAULT_INPUTS = {
    "baseline": "stanford_baseline_batch.json",
    "no_pose": "stanford_no_pose_pipeline_fixed_batch.json",
    "no_robust": "stanford_no_robust_vote_batch.json",
    "no_rsmrq": "stanford_no_rsmrq_batch.json",
    "ours": "stanford_retrieval_cached_batch.json",
}

METHOD_ORDER = ["baseline", "no_pose", "no_robust", "no_rsmrq", "ours"]

METRIC_ALIASES = {
    "add": [("metrics", "ADD"), (None, "ADD"), (None, "add")],
    "add_s": [("metrics", "ADD_S"), ("metrics", "ADD-S"), (None, "ADD_S"), (None, "add_s")],
    "rot": [("metrics", "rotation_error_deg"), (None, "rotation_error_deg"), (None, "rot")],
    "trans": [("metrics", "translation_error"), (None, "translation_error"), (None, "trans")],
    "time": [
        ("stats", "registration_time"),
        ("stats", "total_time"),
        (None, "registration_time"),
        (None, "total_time"),
        (None, "time"),
    ],
}

OUTPUT_COLUMNS = [
    "method",
    "n",
    "add_mean", "add_median", "add_std", "add_q1", "add_q3", "add_iqr",
    "add_s_mean", "add_s_median", "add_s_std", "add_s_q1", "add_s_q3", "add_s_iqr",
    "rot_mean", "rot_median", "rot_std", "rot_q1", "rot_q3", "rot_iqr",
    "trans_mean", "trans_median", "trans_std", "trans_q1", "trans_q3", "trans_iqr",
    "time_mean", "time_median", "time_std", "time_q1", "time_q3", "time_iqr",
    "add_success", "add_s_success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build summary_stats.csv from ablation/baseline result JSON files."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Method/file pair in the form method=path.json. Can be repeated. "
            "If omitted, built-in Stanford defaults are used."
        ),
    )
    parser.add_argument(
        "--output",
        default="summary_stats.csv",
        help="Output CSV path. Default: summary_stats.csv",
    )
    parser.add_argument(
        "--add-threshold",
        type=float,
        default=0.02,
        help="Success threshold for ADD. Default: 0.02",
    )
    parser.add_argument(
        "--add-s-threshold",
        type=float,
        default=0.02,
        help="Success threshold for ADD-S. Default: 0.02",
    )
    return parser.parse_args()



def resolve_inputs(cli_inputs: List[str], output_path: Path) -> Dict[str, Path]:
    base_dir = output_path.resolve().parent
    if not cli_inputs:
        return {k: (base_dir / v).resolve() for k, v in DEFAULT_INPUTS.items()}

    resolved: Dict[str, Path] = {}
    for item in cli_inputs:
        if "=" not in item:
            raise ValueError(f"Invalid --input value: {item!r}. Expected method=path.json")
        method, raw_path = item.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Invalid method name in --input: {item!r}")
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        resolved[method] = path
    return resolved



def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unsupported JSON format in {path}")



def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val



def extract_value(record: dict, aliases: Iterable[tuple]) -> Optional[float]:
    for section, key in aliases:
        source = record
        if section is not None:
            source = record.get(section, {})
            if not isinstance(source, dict):
                continue
        if key in source:
            val = coerce_float(source.get(key))
            if val is not None:
                return val
    return None



def summarize_array(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    if arr.size == 0:
        raise ValueError(f"No valid samples found for {prefix}")

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": float(q3 - q1),
    }



def build_row(method: str, path: Path, add_threshold: float, add_s_threshold: float) -> Dict[str, float]:
    records = load_records(path)
    extracted: Dict[str, List[float]] = {k: [] for k in METRIC_ALIASES}

    for record in records:
        if not isinstance(record, dict):
            continue
        for prefix, aliases in METRIC_ALIASES.items():
            value = extract_value(record, aliases)
            if value is not None:
                extracted[prefix].append(value)

    row: Dict[str, float] = {"method": method}

    add_arr = np.asarray(extracted["add"], dtype=float)
    add_s_arr = np.asarray(extracted["add_s"], dtype=float)
    rot_arr = np.asarray(extracted["rot"], dtype=float)
    trans_arr = np.asarray(extracted["trans"], dtype=float)
    time_arr = np.asarray(extracted["time"], dtype=float)

    row["n"] = int(add_arr.size)
    row.update(summarize_array(add_arr, "add"))
    row.update(summarize_array(add_s_arr, "add_s"))
    row.update(summarize_array(rot_arr, "rot"))
    row.update(summarize_array(trans_arr, "trans"))
    row.update(summarize_array(time_arr, "time"))
    row["add_success"] = float(np.mean(add_arr < add_threshold))
    row["add_s_success"] = float(np.mean(add_s_arr < add_s_threshold))

    return row



def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    inputs = resolve_inputs(args.input, output_path)

    rows: List[Dict[str, float]] = []
    missing = [f"{method}: {path}" for method, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    ordered_methods = [m for m in METHOD_ORDER if m in inputs] + [m for m in inputs if m not in METHOD_ORDER]

    for method in ordered_methods:
        path = inputs[method]
        row = build_row(method, path, args.add_threshold, args.add_s_threshold)
        rows.append(row)
        print(
            f"[OK] {method:<10} n={row['n']:<3d} "
            f"ADD(mean/median)={row['add_mean']:.6f}/{row['add_median']:.6f} "
            f"Time(mean/median)={row['time_mean']:.3f}/{row['time_median']:.3f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved: {output_path}")
    print(f"ADD success threshold: {args.add_threshold}")
    print(f"ADD-S success threshold: {args.add_s_threshold}")


if __name__ == "__main__":
    main()
