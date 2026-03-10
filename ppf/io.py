from typing import Any, Dict, Optional
import open3d as o3d

from .utils import load_yaml


def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    return pcd


def load_config(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if cfg is None:
        cfg = {}
    return cfg
