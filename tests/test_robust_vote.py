# 导入 unittest 模块，用于编写和运行单元测试。
import unittest
# 导入 numpy 并命名为 np，用于构造测试数组。
import numpy as np

# 从 ppf.voting_robust 模块导入 RobustVoter 类。
from ppf.voting_robust import RobustVoter


# 定义鲁棒投票模块的测试用例类。
class TestRobustVote(unittest.TestCase):
    # 测试不同核函数下的权重计算行为是否符合预期。
    def test_kernels(self):
        # 构造初始测试配置，先使用截断核。
        cfg = {
            "kernel": "trunc",
            "tau": 1.0,
            "sigma": 0.5,
            "B": 1.0,
            "top_m_per_bucket": 10,
            "normalize_feature": True
        }
        # 创建鲁棒投票器实例。
        v = RobustVoter(cfg, logger=None)
        # 断言在阈值范围内时，截断核权重为 1。
        self.assertEqual(v.compute_weight(0.5), 1.0)
        # 断言超过阈值后，截断核权重为 0。
        self.assertEqual(v.compute_weight(1.5), 0.0)

        # 切换为高斯核。
        cfg["kernel"] = "gaussian"
        # 基于新配置重新创建投票器。
        v = RobustVoter(cfg, logger=None)
        # 计算零残差下的权重。
        w0 = v.compute_weight(0.0)
        # 计算较大残差下的权重。
        w1 = v.compute_weight(1.0)
        # 断言高斯核在零残差下的权重大于等于较大残差时的权重。
        self.assertTrue(w0 >= w1)

        # 切换为 Huber 核。
        cfg["kernel"] = "huber"
        # 重新创建投票器。
        v = RobustVoter(cfg, logger=None)
        # 断言小残差时 Huber 核权重近似为 1。
        self.assertAlmostEqual(v.compute_weight(0.5), 1.0, places=6)
        # 断言大残差时 Huber 核权重小于 1。
        self.assertTrue(v.compute_weight(2.0) < 1.0)

        # 切换为 Tukey 核。
        cfg["kernel"] = "tukey"
        # 重新创建投票器。
        v = RobustVoter(cfg, logger=None)
        # 断言较小残差时 Tukey 核权重大于 0。
        self.assertTrue(v.compute_weight(0.5) > 0.0)
        # 断言很大残差时 Tukey 核权重为 0。
        self.assertEqual(v.compute_weight(10.0), 0.0)

    # 测试残差函数能够返回正值。
    def test_residual(self):
        # 构造高斯核下的测试配置。
        cfg = {"kernel": "gaussian", "sigma": 1.0, "tau": 1.0, "B": 1.0, "top_m_per_bucket": 10, "normalize_feature": True}
        # 创建鲁棒投票器实例。
        v = RobustVoter(cfg, logger=None)
        # 构造场景特征向量。
        gs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # 构造模型特征向量。
        gm = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        # 计算两者在给定模型直径下的残差。
        r = v.residual(gs, gm, model_diameter=10.0)
        # 断言残差应大于 0。
        self.assertTrue(r > 0.0)


# 当测试文件被直接运行时，启动 unittest 测试入口。
if __name__ == "__main__":
    # 运行当前文件中的全部单元测试。
    unittest.main()
