import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np


@dataclass(frozen=True)
class PPFEntry:
    mr: int
    mi: int
    g: Optional[Tuple[float, float, float, float]]  # internal feature g (optional to save memory)


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
        assert merge_mode in ("union", "count")
        self.w_levels = [np.array(w, dtype=np.float32) for w in w_levels]
        self.L = len(self.w_levels)
        self.T = int(T_tables)
        self.merge_mode = merge_mode
        self.seed = seed
        self.logger = logger

        rng = np.random.RandomState(seed)
        # offsets[level][t] = u vector
        self.offsets: List[List[np.ndarray]] = []
        for l in range(self.L):
            wl = self.w_levels[l]
            level_offsets = []
            for _ in range(self.T):
                u = rng.uniform(low=0.0, high=wl).astype(np.float32)
                level_offsets.append(u)
            self.offsets.append(level_offsets)

        # tables[level][t]: dict[key]->list[PPFEntry]
        self.tables: List[List[Dict[Tuple[int, int, int, int], List[PPFEntry]]]] = [
            [dict() for _ in range(self.T)] for __ in range(self.L)
        ]

        if self.logger:
            self.logger.info("[RS-MRQ] Initialized.")
            for l in range(self.L):
                self.logger.info(f"[RS-MRQ] Level {l}: w={self.w_levels[l].tolist()}")
                for t in range(self.T):
                    self.logger.info(f"[RS-MRQ] Level {l} Table {t}: u={self.offsets[l][t].tolist()}")
            self.logger.info(f"[RS-MRQ] merge_mode={self.merge_mode}, seed={self.seed}")

    @staticmethod
    def _quantize(g: np.ndarray, w: np.ndarray, u: np.ndarray) -> Tuple[int, int, int, int]:
        q = np.floor((g + u) / w).astype(np.int64)
        return (int(q[0]), int(q[1]), int(q[2]), int(q[3]))

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

    def query_buckets(self, g: np.ndarray) -> List[List[PPFEntry]]:
        """
        Returns list of buckets (each bucket is a list of PPFEntry).
        """
        buckets: List[List[PPFEntry]] = []
        for l in range(self.L):
            w = self.w_levels[l]
            for t in range(self.T):
                u = self.offsets[l][t]
                key = self._quantize(g, w, u)
                b = self.tables[l][t].get(key, [])
                buckets.append(b)
        return buckets

    def merge_candidates(
        self,
        buckets: List[List[PPFEntry]]
    ) -> Dict[Tuple[int, int], Tuple[PPFEntry, int]]:
        """
        Merge by union or count:
        returns dict[(mr,mi)] -> (entry, count)
        """
        out: Dict[Tuple[int, int], Tuple[PPFEntry, int]] = {}
        for b in buckets:
            for e in b:
                k = (e.mr, e.mi)
                if k not in out:
                    out[k] = (e, 1)
                else:
                    ee, c = out[k]
                    out[k] = (ee, c + 1)
        if self.merge_mode == "union":
            # force count=1
            out = {k: (v[0], 1) for k, v in out.items()}
        return out

    @staticmethod
    def bucket_stats(buckets: List[List[PPFEntry]]) -> Tuple[int, int]:
        nb = len(buckets)
        tot = sum(len(b) for b in buckets)
        return nb, tot
