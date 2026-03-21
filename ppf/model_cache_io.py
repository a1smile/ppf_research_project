# model_cache_io.py
# 中文说明：该模块负责 PPF 模型缓存的保存、加载以及一致性校验。
# 作用：保存/加载 PPFModel 缓存（一次构建，反复使用）
from __future__ import annotations

# 导入 os、pickle、hashlib，用于路径处理、对象序列化和哈希计算。
import os
import pickle
import hashlib
# 导入 dataclass 装饰器和类型标注工具。
from dataclasses import dataclass
from typing import Any, Dict, Optional

# 导入目录创建工具函数。
from .utils import ensure_dir


# 定义缓存元信息结构，用于记录影响缓存有效性的关键配置。
@dataclass
class CacheMeta:
    """
    用于校验：缓存模型是否和当前 cfg / step 参数一致
    """
    # 缓存格式版本号。
    cache_version: str
    # 模型名称。
    model_name: str
    # 角度步长（度）。
    angle_step_deg: float
    # 模型下采样体素大小。
    sampling_leaf: float
    # 法向量估计使用的近邻数。
    normal_k: int
    # 距离步长比例。
    distance_step_ratio: float
    # 是否启用 RS-MRQ。
    enable_rsmrq: bool
    # 是否启用鲁棒投票。
    enable_robust_vote: bool
    # RS-MRQ 相关配置。
    rsmrq_cfg: Dict[str, Any]
    # RobustVote 相关配置。
    robust_vote_cfg: Dict[str, Any]

    # 根据当前元信息生成稳定指纹，用于与缓存文件或当前配置比对。
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
        # 返回 SHA-256 十六进制摘要。
        return hashlib.sha256(s).hexdigest()


# 定义缓存元信息不匹配时抛出的专用异常。
class CacheMetaMismatchError(RuntimeError):
    pass


# 根据模型名和配置构造对应的缓存元信息对象。
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


# 将 PPF 模型和元信息写入缓存文件。
def save_ppf_model_cache(cache_path: str, ppf_model: Any, meta: CacheMeta) -> str:
    # 确保缓存文件所在目录存在。
    ensure_dir(os.path.dirname(cache_path) or ".")
    # 组织待序列化的数据载荷。
    payload = {
        "meta": meta,
        "fingerprint": meta.fingerprint(),
        "ppf_model": ppf_model,
    }
    # 以二进制 pickle 形式写入文件。
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 返回缓存路径，便于上层链式使用。
    return cache_path


# 从缓存文件加载 PPF 模型，并按需校验当前配置是否一致。
def load_ppf_model_cache(
    cache_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    strict: bool = True
) -> Any:
    """
    strict=True：cfg 和缓存 meta 不一致就报错（推荐）
    strict=False：不一致也加载（你自己承担风险）
    """
    # 读取并反序列化缓存文件。
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    # 取出缓存元信息、保存时指纹和模型对象。
    meta: CacheMeta = payload["meta"]
    fp_saved: str = payload["fingerprint"]
    model = payload["ppf_model"]

    # 先校验文件中保存的指纹是否与元信息当前计算结果一致。
    if fp_saved != meta.fingerprint():
        # 文件被修改/损坏
        raise CacheMetaMismatchError(f"缓存文件校验失败：{cache_path}")

    # 若传入了当前配置且启用严格模式，则进一步校验缓存与当前配置是否匹配。
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

    # 返回缓存中的模型对象。
    return model
