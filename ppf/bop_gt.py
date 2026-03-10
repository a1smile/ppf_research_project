import os
import json
from functools import lru_cache
from typing import Any, Dict, Tuple, Optional

import numpy as np


@lru_cache(maxsize=256)
def _load_json_cached(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scene_dir_from_depth_path(depth_path: str) -> str:
    """
    Given .../scene_id/depth/000000.png -> returns .../scene_id
    Works for Windows paths as well.
    """
    depth_dir = os.path.dirname(depth_path)
    scene_dir = os.path.dirname(depth_dir)
    return scene_dir


def bop_pose_from_scene_gt_entry(entry: Dict[str, Any], t_scale: float = 1.0) -> np.ndarray:
    """
    BOP scene_gt.json entry:
      - cam_R_m2c: 9 numbers (row-major 3x3)
      - cam_t_m2c: 3 numbers (usually in mm for BOP datasets)
    t_scale:
      - if your point clouds are in mm: t_scale=1.0
      - if your point clouds are in meters: t_scale=0.001
    """
    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(entry["cam_t_m2c"], dtype=np.float64).reshape(3) * float(t_scale)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_bop_gt_pose(
    scene_dir: str,
    frame_id: int,
    gt_id: int,
    t_scale: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """
    Returns:
      (obj_id, T_gt)
    where:
      - obj_id is the BOP object category id (used to locate model)
      - T_gt maps model -> camera coordinates
    """
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing scene_gt.json at: {gt_path}")

    scene_gt = _load_json_cached(gt_path)

    key = str(int(frame_id))  # BOP uses integer string keys
    if key not in scene_gt:
        raise KeyError(f"frame_id={frame_id} not found in {gt_path}")

    gt_list = scene_gt[key]
    if gt_id < 0 or gt_id >= len(gt_list):
        raise IndexError(f"gt_id={gt_id} out of range for frame_id={frame_id}, len={len(gt_list)}")

    entry = gt_list[int(gt_id)]
    obj_id = int(entry["obj_id"])
    T_gt = bop_pose_from_scene_gt_entry(entry, t_scale=t_scale)
    return obj_id, T_gt


def try_get_bop_gt_pose(
    scene_dir: str,
    frame_id: int,
    gt_id: int,
    t_scale: float = 1.0,
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[str]]:
    """
    Safe wrapper: never throws; returns (obj_id, T_gt, error_message).
    """
    try:
        obj_id, T = get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=t_scale)
        return obj_id, T, None
    except Exception as e:
        return None, None, str(e)
