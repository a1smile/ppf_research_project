# build_model_cache.py
# 运行方式（重要）：在“该文件夹的上一级目录”执行：
#   python -m <文件夹名>.build_model_cache
#
# 例如你的文件夹叫 ppf_pkg，那么：
#   python -m ppf_pkg.build_model_cache

from __future__ import annotations

import os
import math
import ast
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp

# ✅ 同包相对导入（替换掉 your_package.*）
from .model_builder import build_ppf_model
from .model_cache_io import save_ppf_model_cache, make_cache_meta
from .preprocess import subsample_and_calculate_normals_model


ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "model_data.csv")
OUT_DIR = os.path.join(ROOT, "model_cache")


def _parse_cell(x):
    """
    points/normals 在 csv 里通常是字符串形式的 list
    优先 json，再退回 ast.literal_eval
    """
    if isinstance(x, (list, tuple)):
        return x
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def _build_one(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, str]:
    model_name = str(row["model_name"])
    pts = np.asarray(_parse_cell(row["points"]), dtype=np.float64)
    normals = np.asarray(_parse_cell(row["normals"]), dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{model_name}: points 形状不对: {pts.shape}")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"{model_name}: normals 形状不对: {normals.shape}")
    if pts.shape[0] != normals.shape[0]:
        raise ValueError(f"{model_name}: points/normals 数量不一致: {pts.shape[0]} vs {normals.shape[0]}")

    # 构造 open3d 点云（用你已有的模型数据，不再读 ply）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 使用和在线验证一致的预处理参数
    sampling_leaf = float(cfg.get("sampling_leaf", 5.0))
    normal_k = int(cfg.get("normal_k", 5))
    model_down = subsample_and_calculate_normals_model(pcd, voxel_size=sampling_leaf, k=normal_k)

    # 计算 angle_step / distance_step（必须和 registration.py 一致）
    angle_step_deg = float(cfg.get("angle_step_deg", 12.0))
    angle_step = math.radians(angle_step_deg)
    distance_step = float(cfg.get("distance_step_ratio", 0.6)) * sampling_leaf

    # 构建 PPFModel（最耗时 O(N^2) 只做一次）
    ppf_model = build_ppf_model(model_down, angle_step, distance_step, cfg, logger=None)

    # 保存缓存
    meta = make_cache_meta(model_name, cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{os.path.splitext(model_name)[0]}_ppf.pkl")
    save_ppf_model_cache(out_path, ppf_model, meta)
    return model_name, out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 这里的 cfg 必须与你在线验证时一致（否则缓存校验会报错）
    cfg = {
        "seed": 0,
        "sampling_leaf": 5.0,
        "normal_k": 5,
        "angle_step_deg": 12.0,
        "distance_step_ratio": 0.6,
        "enable_rsmrq": False,
        "enable_robust_vote": False,
        "rsmrq": {},
        "robust_vote": {},
    }

    df = pd.read_csv(CSV_PATH)
    rows = df.to_dict(orient="records")
    if not rows:
        raise RuntimeError(f"CSV 为空：{CSV_PATH}")

    # Windows 多进程：spawn
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # i9-14900HX：建议 8~16
    workers = min(16, os.cpu_count() or 8)

    print(f"[CacheBuild] models={len(rows)} workers={workers}")
    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_build_one, row, cfg) for row in rows]
        for fut in as_completed(futs):
            try:
                model_name, out_path = fut.result()
                ok += 1
                print(f"[OK] {model_name} -> {out_path}")
            except Exception as e:
                print(f"[FAIL] {e}")

    print(f"[Done] success={ok}/{len(rows)} cache_dir={OUT_DIR}")


if __name__ == "__main__":
    main()