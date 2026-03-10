# model_cache_io.py
# 作用：保存/加载 PPFModel 缓存（一次构建，反复使用）
from __future__ import annotations

import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils import ensure_dir


@dataclass
class CacheMeta:
    """
    用于校验：缓存模型是否和当前 cfg / step 参数一致
    """
    cache_version: str
    model_name: str
    angle_step_deg: float
    sampling_leaf: float
    normal_k: int
    distance_step_ratio: float
    enable_rsmrq: bool
    enable_robust_vote: bool
    rsmrq_cfg: Dict[str, Any]
    robust_vote_cfg: Dict[str, Any]

    def fingerprint(self) -> str:
        # 生成可复现的 hash，用于校验一致性
        # 注意：只把“会影响模型点对库”的参数纳入 fingerprint
        s = repr({
            "cache_version": self.cache_version,
            "model_name": self.model_name,
            "angle_step_deg": float(self.angle_step_deg),
            "sampling_leaf": float(self.sampling_leaf),
            "normal_k": int(self.normal_k),
            "distance_step_ratio": float(self.distance_step_ratio),
            "enable_rsmrq": bool(self.enable_rsmrq),
            "enable_robust_vote": bool(self.enable_robust_vote),
            "rsmrq_cfg": self.rsmrq_cfg or {},
            "robust_vote_cfg": self.robust_vote_cfg or {},
        }).encode("utf-8")
        return hashlib.sha256(s).hexdigest()


class CacheMetaMismatchError(RuntimeError):
    pass


def make_cache_meta(model_name: str, cfg: Dict[str, Any]) -> CacheMeta:
    return CacheMeta(
        cache_version="ppf_cache_v1",
        model_name=str(model_name),
        angle_step_deg=float(cfg.get("angle_step_deg", 12.0)),
        sampling_leaf=float(cfg.get("sampling_leaf", 5.0)),
        normal_k=int(cfg.get("normal_k", 5)),
        distance_step_ratio=float(cfg.get("distance_step_ratio", 0.6)),
        enable_rsmrq=bool(cfg.get("enable_rsmrq", False)),
        enable_robust_vote=bool(cfg.get("enable_robust_vote", False)),
        rsmrq_cfg=dict(cfg.get("rsmrq", {}) or {}),
        robust_vote_cfg=dict(cfg.get("robust_vote", {}) or {}),
    )


def save_ppf_model_cache(cache_path: str, ppf_model: Any, meta: CacheMeta) -> str:
    ensure_dir(os.path.dirname(cache_path) or ".")
    payload = {
        "meta": meta,
        "fingerprint": meta.fingerprint(),
        "ppf_model": ppf_model,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_ppf_model_cache(
    cache_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    strict: bool = True
) -> Any:
    """
    strict=True：cfg 和缓存 meta 不一致就报错（推荐）
    strict=False：不一致也加载（你自己承担风险）
    """
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    meta: CacheMeta = payload["meta"]
    fp_saved: str = payload["fingerprint"]
    model = payload["ppf_model"]

    if fp_saved != meta.fingerprint():
        # 文件被修改/损坏
        raise CacheMetaMismatchError(f"缓存文件校验失败：{cache_path}")

    if cfg is not None and strict:
        meta_now = make_cache_meta(meta.model_name, cfg)
        if meta_now.fingerprint() != meta.fingerprint():
            raise CacheMetaMismatchError(
                "缓存模型与当前 cfg 不一致，拒绝加载。\n"
                f"cache={cache_path}\n"
                f"cache_fingerprint={meta.fingerprint()}\n"
                f"cfg_fingerprint={meta_now.fingerprint()}\n"
                "请用相同 cfg 重新生成缓存，或 strict=False 强行加载。"
            )

    return model