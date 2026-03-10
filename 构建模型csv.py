import open3d as o3d
import numpy as np
import pandas as pd
import os


def load_ply_files(ply_directory):
    """
    从指定的目录加载所有的 .ply 文件
    """
    ply_files = [f for f in os.listdir(ply_directory) if f.endswith('.ply')]
    ply_file_paths = [os.path.join(ply_directory, f) for f in ply_files]
    return ply_file_paths


def process_ply_file(ply_file_path):
    """
    处理单个 .ply 文件，提取点云数据和法线
    """
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # 获取点云数据和法线
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    return points, normals


def build_model_csv(ply_directory, output_csv="model_data.csv"):
    """
    从给定目录加载 .ply 文件，计算每个点云的相关属性，并保存为 CSV 文件
    """
    # 获取所有 .ply 文件
    ply_file_paths = load_ply_files(ply_directory)

    model_data = []

    for ply_file_path in ply_file_paths:
        points, normals = process_ply_file(ply_file_path)

        # 获取模型的基本信息
        model_name = os.path.basename(ply_file_path)
        model_diameter = np.max(np.linalg.norm(points - points.mean(axis=0), axis=1))

        # 这里你可以根据需求添加更多计算，例如 alpha_m 等
        # 对每个点云数据执行计算或提取特征

        # 示例：将点云数据存储为平面化的形式
        model_data.append({
            "model_name": model_name,
            "model_diameter": model_diameter,
            "points": points.tolist(),  # 可根据需要修改为其他特征
            "normals": normals.tolist()
        })

    # 将数据保存到 CSV 文件
    df = pd.DataFrame(model_data)
    df.to_csv(output_csv, index=False)
    print(f"模型数据已保存到 {output_csv}")


# 设置目录路径
ply_directory = r"F:\Research\daihongsong\data\LM-O (Linemod-Occluded)\lmo_models\models_eval"

# 执行模型构建与保存
build_model_csv(ply_directory, output_csv="model_data.csv")
