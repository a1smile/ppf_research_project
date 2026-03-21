# 导入类型标注工具。
from typing import Any, Dict, Optional
# 导入 open3d，用于读取点云文件。
import open3d as o3d

# 从工具模块导入 YAML 加载函数。
from .utils import load_yaml


# 读取点云文件，并检查结果是否为空。
def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    # 调用 Open3D 读取点云。
    pcd = o3d.io.read_point_cloud(path)
    # 如果点云为空，则抛出异常。
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    # 返回读取到的点云对象。
    return pcd


# 读取配置文件，若内容为空则返回空字典。
def load_config(path: str) -> Dict[str, Any]:
    # 加载 YAML 配置内容。
    cfg = load_yaml(path)
    # 若 YAML 返回 None，则转为空字典。
    if cfg is None:
        cfg = {}
    # 返回配置字典。
    return cfg
