# 导入 logging、dataclass 和类型标注工具。
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

# 导入 numpy 并命名为 np，用于数值计算。
import numpy as np


# 定义哈希表中存储的 PPF 条目结构。
@dataclass(frozen=True)
class PPFEntry:
    # 模型参考点索引。
    mr: int
    # 模型配对点索引。
    mi: int
    # 内部特征 g，可选保存以节省内存。
    g: Optional[Tuple[float, float, float, float]]  # internal feature g (optional to save memory)


# RS-MRQ 多层多表哈希结构。
class RSMRQHashTable:
    """
    RS–MRQ Hashing:
    - L levels (w_levels)
    - T tables per level
    - random shifts u ~ Uniform([0,w))
    - query returns bucket lists (one per (level,table)) + stats
    """
    def __init__(
        self,
        w_levels: List[List[float]],
        T_tables: int,
        merge_mode: str = "union",
        seed: int = 0,
        logger: Optional[logging.Logger] = None
    ):
        # 合并方式必须是 union 或 count。
        assert merge_mode in ("union", "count")
        # 保存每层的窗口参数。
        self.w_levels = [np.array(w, dtype=np.float32) for w in w_levels]
        # 层数。
        self.L = len(self.w_levels)
        # 每层哈希表数量。
        self.T = int(T_tables)
        # 保存合并方式、随机种子和日志器。
        self.merge_mode = merge_mode
        self.seed = seed
        self.logger = logger

        # 基于给定种子创建随机数发生器。
        rng = np.random.RandomState(seed)
        # 为每层每张表生成随机偏移向量。
        self.offsets: List[List[np.ndarray]] = []
        for l in range(self.L):
            wl = self.w_levels[l]
            level_offsets = []
            for _ in range(self.T):
                u = rng.uniform(low=0.0, high=wl).astype(np.float32)
                level_offsets.append(u)
            self.offsets.append(level_offsets)

        # 初始化实际的多层多表存储结构。
        self.tables: List[List[Dict[Tuple[int, int, int, int], List[PPFEntry]]]] = [
            [dict() for _ in range(self.T)] for __ in range(self.L)
        ]

        # 若存在日志器，则输出初始化信息。
        if self.logger:
            self.logger.info("[RS-MRQ] Initialized.")
            for l in range(self.L):
                self.logger.info(f"[RS-MRQ] Level {l}: w={self.w_levels[l].tolist()}")
                for t in range(self.T):
                    self.logger.info(f"[RS-MRQ] Level {l} Table {t}: u={self.offsets[l][t].tolist()}")
            self.logger.info(f"[RS-MRQ] merge_mode={self.merge_mode}, seed={self.seed}")

    # 使用窗口参数和偏移量对内部特征 g 进行量化。
    @staticmethod
    def _quantize(g: np.ndarray, w: np.ndarray, u: np.ndarray) -> Tuple[int, int, int, int]:
        q = np.floor((g + u) / w).astype(np.int64)
        return (int(q[0]), int(q[1]), int(q[2]), int(q[3]))

    # 将一个条目插入所有层和所有表对应的桶中。
    def add(self, g: np.ndarray, entry: PPFEntry) -> None:
        for l in range(self.L):
            w = self.w_levels[l]
            for t in range(self.T):
                u = self.offsets[l][t]
                key = self._quantize(g, w, u)
                bucket = self.tables[l][t].get(key)
                if bucket is None:
                    self.tables[l][t][key] = [entry]
                else:
                    bucket.append(entry)

    # 查询给定特征在所有层和所有表中的候选桶。
    def query_buckets(self, g: np.ndarray) -> List[List[PPFEntry]]:
        """
        Returns list of buckets (each bucket is a list of PPFEntry).
        """
        # 用于保存所有查询到的桶。
        buckets: List[List[PPFEntry]] = []
        for l in range(self.L):
            w = self.w_levels[l]
            for t in range(self.T):
                u = self.offsets[l][t]
                key = self._quantize(g, w, u)
                b = self.tables[l][t].get(key, [])
                buckets.append(b)
        # 返回所有桶组成的列表。
        return buckets

    # 将多个桶中的候选条目按 union 或 count 方式合并。
    def merge_candidates(
        self,
        buckets: List[List[PPFEntry]]
    ) -> Dict[Tuple[int, int], Tuple[PPFEntry, int]]:
        """
        Merge by union or count:
        returns dict[(mr,mi)] -> (entry, count)
        """
        # 用于保存合并后的结果。
        out: Dict[Tuple[int, int], Tuple[PPFEntry, int]] = {}
        for b in buckets:
            for e in b:
                k = (e.mr, e.mi)
                if k not in out:
                    out[k] = (e, 1)
                else:
                    ee, c = out[k]
                    out[k] = (ee, c + 1)
        # 若为 union 模式，则强制所有计数为 1。
        if self.merge_mode == "union":
            out = {k: (v[0], 1) for k, v in out.items()}
        # 返回合并后的候选字典。
        return out

    # 统计桶数量和所有桶内条目总数。
    @staticmethod
    def bucket_stats(buckets: List[List[PPFEntry]]) -> Tuple[int, int]:
        nb = len(buckets)
        tot = sum(len(b) for b in buckets)
        return nb, tot
