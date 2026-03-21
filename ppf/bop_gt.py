# 导入 os 模块，用于路径处理。
import os
# 导入 json 模块，用于读取 BOP 标注文件。
import json
# 导入 lru_cache 装饰器，用于缓存重复读取的 JSON 内容。
from functools import lru_cache
# 导入类型标注工具。
from typing import Any, Dict, Tuple, Optional

# 导入 numpy 并命名为 np，用于矩阵和向量计算。
import numpy as np


# 使用 LRU 缓存读取 JSON 文件，避免同一文件被重复解析。
@lru_cache(maxsize=256)
def _load_json_cached(path: str) -> Dict[str, Any]:
    # 以 UTF-8 编码打开 JSON 文件。
    with open(path, "r", encoding="utf-8") as f:
        # 返回解析后的 JSON 对象。
        return json.load(f)


# 根据深度图路径推断对应的场景目录。
def scene_dir_from_depth_path(depth_path: str) -> str:
    """
    Given .../scene_id/depth/000000.png -> returns .../scene_id
    Works for Windows paths as well.
    """
    # 获取 depth 文件所在目录。
    depth_dir = os.path.dirname(depth_path)
    # 再向上一级得到场景目录。
    scene_dir = os.path.dirname(depth_dir)
    # 返回场景目录路径。
    return scene_dir


# 将 BOP 的 scene_gt 条目转换成 4x4 位姿矩阵。
def bop_pose_from_scene_gt_entry(entry: Dict[str, Any], t_scale: float = 1.0) -> np.ndarray:
    """
    BOP scene_gt.json entry:
      - cam_R_m2c: 9 numbers (row-major 3x3)
      - cam_t_m2c: 3 numbers (usually in mm for BOP datasets)
    t_scale:
      - if your point clouds are in mm: t_scale=1.0
      - if your point clouds are in meters: t_scale=0.001
    """
    # 从条目中读取旋转矩阵并重塑为 3x3。
    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    # 从条目中读取平移向量，并按给定比例进行单位缩放。
    t = np.array(entry["cam_t_m2c"], dtype=np.float64).reshape(3) * float(t_scale)

    # 初始化 4x4 单位矩阵作为齐次变换矩阵。
    T = np.eye(4, dtype=np.float64)
    # 填入旋转部分。
    T[:3, :3] = R
    # 填入平移部分。
    T[:3, 3] = t
    # 返回完整的 4x4 位姿矩阵。
    return T


# 从指定场景目录的 BOP 标注中读取某帧某目标的 GT 位姿。
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
    # 构造 scene_gt.json 的完整路径。
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    # 如果文件不存在，则抛出异常。
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing scene_gt.json at: {gt_path}")

    # 读取并缓存 scene_gt.json 内容。
    scene_gt = _load_json_cached(gt_path)

    # BOP 标注中的帧键通常是整数字符串。
    key = str(int(frame_id))  # BOP uses integer string keys
    # 如果指定帧不存在，则抛出异常。
    if key not in scene_gt:
        raise KeyError(f"frame_id={frame_id} not found in {gt_path}")

    # 取出该帧对应的所有 GT 条目列表。
    gt_list = scene_gt[key]
    # 检查 gt_id 是否在有效范围内。
    if gt_id < 0 or gt_id >= len(gt_list):
        raise IndexError(f"gt_id={gt_id} out of range for frame_id={frame_id}, len={len(gt_list)}")

    # 取出指定目标实例的标注条目。
    entry = gt_list[int(gt_id)]
    # 读取其中的 obj_id。
    obj_id = int(entry["obj_id"])
    # 将条目转换为 4x4 GT 位姿矩阵。
    T_gt = bop_pose_from_scene_gt_entry(entry, t_scale=t_scale)
    # 返回对象类别 id 和 GT 位姿。
    return obj_id, T_gt


# 安全包装函数，读取失败时不抛异常而是返回错误信息。
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
        # 尝试调用严格版本的 GT 读取函数。
        obj_id, T = get_bop_gt_pose(scene_dir, frame_id, gt_id, t_scale=t_scale)
        # 成功时返回 obj_id、位姿以及空错误信息。
        return obj_id, T, None
    except Exception as e:
        # 失败时返回空结果和错误字符串。
        return None, None, str(e)
