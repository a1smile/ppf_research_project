import unittest
import numpy as np

from ppf.rsmrq_hash import RSMRQHashTable, PPFEntry


class TestRSMRQ(unittest.TestCase):
    def test_deterministic_offsets(self):
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        t1 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)
        t2 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)

        for i in range(2):
            self.assertTrue(np.allclose(t1.offsets[0][i], t2.offsets[0][i]))

    def test_query_retrieves_added(self):
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        ht = RSMRQHashTable(w_levels=w_levels, T_tables=1, merge_mode="union", seed=0, logger=None)
        g = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        e = PPFEntry(mr=1, mi=2, g=(0.2, 0.2, 0.2, 0.2))
        ht.add(g, e)

        buckets = ht.query_buckets(g)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(len(buckets[0]), 1)
        self.assertEqual(buckets[0][0].mr, 1)
        self.assertEqual(buckets[0][0].mi, 2)


if __name__ == "__main__":
    unittest.main()
