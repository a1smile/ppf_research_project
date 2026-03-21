# 导入 unittest 模块，用于编写和运行单元测试。
import unittest
# 导入 numpy 并命名为 np，用于构造测试数组。
import numpy as np

# 从 ppf.kde_refine 模块导入 KDEMeanShiftRefiner 类。
from ppf.kde_refine import KDEMeanShiftRefiner


# 定义 KDE 精化模块的测试用例类。
class TestKDERefine(unittest.TestCase):
    # 测试 refine 方法能够正常运行并返回合理结果。
    def test_refine_runs(self):
        # 构造 KDE 精化器的测试配置。
        cfg = {
            "use_angle_embedding": True,
            "top_k": 3,
            "bandwidth_h": 0.3,
            "max_iter": 50,
            "tol": 1e-6
        }
        # 创建 KDE 均值漂移精化器实例，不传日志器。
        r = KDEMeanShiftRefiner(cfg, logger=None)
        # 构造带有角度环绕特性的样本角度数组。
        thetas = np.array([0.0, 0.05, -0.03, 3.12, -3.10], dtype=np.float32)  # wrap-around samples
        # 为每个样本设置权重。
        weights = np.array([1, 1, 1, 0.2, 0.2], dtype=np.float32)
        # 调用 refine 方法，获得精化后的角度和迭代轨迹。
        theta_ref, trace = r.refine(thetas, weights, theta_init=0.0)
        # 断言精化后的角度应接近 0。
        self.assertTrue(abs(theta_ref) < 0.2)
        # 断言至少执行过一次迭代。
        self.assertTrue(trace.iters >= 1)

    # 测试 KDE 轨迹中的核密度值近似单调不减。
    def test_monotonic_trace(self):
        # 构造另一组测试配置。
        cfg = {"use_angle_embedding": True, "bandwidth_h": 0.4, "max_iter": 20, "tol": 1e-8}
        # 创建 KDE 均值漂移精化器实例。
        r = KDEMeanShiftRefiner(cfg, logger=None)
        # 构造一组彼此接近的角度样本。
        thetas = np.array([0.0, 0.1, 0.12, 0.08], dtype=np.float32)
        # 构造与角度数组同形状的全 1 权重数组。
        weights = np.ones_like(thetas)
        # 执行 refine，并获取迭代轨迹。
        _, trace = r.refine(thetas, weights, theta_init=0.05)
        # 经验性检查核密度值是否近似单调不减。
        vals = trace.kde_values
        # 从第二个值开始逐个比较前后两次的核密度值。
        for i in range(1, len(vals)):
            # 断言当前值不小于前一个值太多，允许极小数值误差。
            self.assertTrue(vals[i] >= vals[i-1] - 1e-6)


# 当测试文件被直接运行时，启动 unittest 测试入口。
if __name__ == "__main__":
    # 运行当前文件中的全部单元测试。
    unittest.main()
