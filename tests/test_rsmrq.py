# 导入 unittest 模块，用于编写和运行单元测试。
import unittest
# 导入 numpy 并命名为 np，用于数值比较和构造测试数组。
import numpy as np

# 从 ppf.rsmrq_hash 模块导入哈希表类和条目类。
from ppf.rsmrq_hash import RSMRQHashTable, PPFEntry


# 定义 RS-MRQ 哈希模块的测试用例类。
class TestRSMRQ(unittest.TestCase):
    # 测试在相同随机种子下，各哈希表的偏移量应保持确定性一致。
    def test_deterministic_offsets(self):
        # 构造仅包含一组窗口参数的测试配置。
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        # 使用相同参数和随机种子创建第一个哈希表对象。
        t1 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)
        # 使用相同参数和随机种子创建第二个哈希表对象。
        t2 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)

        # 逐个比较两个对象在同一层中的偏移量是否一致。
        for i in range(2):
            # 断言对应偏移量数组近似相等。
            self.assertTrue(np.allclose(t1.offsets[0][i], t2.offsets[0][i]))

    # 测试添加条目后，查询能够正确取回已添加的内容。
    def test_query_retrieves_added(self):
        # 构造仅包含一组窗口参数的测试配置。
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        # 创建一个只有单张哈希表的 RS-MRQ 哈希对象。
        ht = RSMRQHashTable(w_levels=w_levels, T_tables=1, merge_mode="union", seed=0, logger=None)
        # 构造一个测试特征向量。
        g = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        # 构造一个待加入哈希表的 PPF 条目。
        e = PPFEntry(mr=1, mi=2, g=(0.2, 0.2, 0.2, 0.2))
        # 将特征和条目加入哈希表。
        ht.add(g, e)

        # 查询与该特征对应的桶内容。
        buckets = ht.query_buckets(g)
        # 断言总共返回一个桶。
        self.assertEqual(len(buckets), 1)
        # 断言该桶内恰好有一个条目。
        self.assertEqual(len(buckets[0]), 1)
        # 断言条目的 mr 字段与插入时一致。
        self.assertEqual(buckets[0][0].mr, 1)
        # 断言条目的 mi 字段与插入时一致。
        self.assertEqual(buckets[0][0].mi, 2)


# 当测试文件被直接运行时，启动 unittest 测试入口。
if __name__ == "__main__":
    # 运行当前文件中的全部单元测试。
    unittest.main()
