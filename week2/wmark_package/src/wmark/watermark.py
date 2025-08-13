import math, hashlib, random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from scipy.stats import norm


def _int_from_hash(*parts: bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest(), "big")


def _rng_from_context(key: bytes, prefix_ids: List[int], pos: int) -> random.Random:
    ctx_hash = _int_from_hash(key, bytes(prefix_ids), pos.to_bytes(4, "big"))
    return random.Random(ctx_hash)


# --------------------------
# 1) SoftWatermark：greenlist + logit偏置
# --------------------------
@dataclass
class SoftWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    max_green_cap: Optional[int] = None  # 限制每步green集合大小（可提速）


class SoftWatermark:
    """
    经典 soft watermark（greenlist + logit 偏置）：
    - 每个位置t：用(key, prefix, t)构造随机greenlist (~ gamma * |V|)
    - 生成时：对greenlist内token加delta偏置
    - 检测：统计命中数做z检
    """
    def __init__(self, tokenizer, cfg: SoftWMConfig):
        self.tok = tokenizer
        self.cfg = cfg
        self.key_bytes = cfg.key.encode("utf-8")

    def _green_ids_at(self, prefix_ids: List[int], pos: int, vocab_size: int) -> List[int]:
        k = int(max(1, math.floor(self.cfg.gamma * vocab_size)))
        if self.cfg.max_green_cap:
            k = min(k, self.cfg.max_green_cap)
        rng = _rng_from_context(self.key_bytes, prefix_ids, pos)
        return rng.sample(range(vocab_size), k)

    def bias_logits(self, logits: torch.Tensor, prefix_ids: List[int], pos: int):
        green = self._green_ids_at(prefix_ids, pos, logits.shape[-1])
        logits[green] += self.cfg.delta

    def count_hits(self, all_ids: List[int], prompt_ids: List[int]) -> Tuple[int, int]:
        vocab_size = self.tok.vocab_size
        n = len(all_ids) - len(prompt_ids)
        C = 0
        for i in range(len(prompt_ids), len(all_ids)):
            pos = i
            prefix = all_ids[:i]
            token_id = all_ids[i]
            green = self._green_ids_at(prefix, pos, vocab_size)
            if token_id in green:
                C += 1
        return C, n

    def z_test(self, C: int, n: int) -> Tuple[float, float]:
        gamma = self.cfg.gamma
        if n == 0 or gamma <= 0 or gamma >= 1:
            return 0.0, 1.0
        mean = n * gamma
        var = n * gamma * (1 - gamma)
        if var <= 0:
            return 0.0, 1.0
        z = (C - mean) / math.sqrt(var)
        p = 1 - norm.cdf(z)
        return z, p


# --------------------------
# 2) HashBucketWatermark：按token哈希到桶
# --------------------------
@dataclass
class HBWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    num_buckets: int = 1024

class HashBucketWatermark:
    """
    可扩展哈希水印（分桶）：
    - (key, prefix, pos) => 绿色桶集合（约 gamma 比例）
    - (key, pos, token_id) => token 所在桶
    - 命中绿色桶则加偏置；检测复现实验设置并做 z 检
    """
    def __init__(self, tokenizer, cfg: HBWMConfig):
        assert 0 < cfg.gamma < 1
        self.tok = tokenizer
        self.cfg = cfg
        self.key_bytes = cfg.key.encode("utf-8")

    def _green_bucket_mask(self, prefix_ids: List[int], pos: int) -> List[bool]:
        rng = _rng_from_context(self.key_bytes, prefix_ids, pos)
        k = max(1, int(self.cfg.gamma * self.cfg.num_buckets))
        green_idx = set(rng.sample(range(self.cfg.num_buckets), k))
        return [i in green_idx for i in range(self.cfg.num_buckets)]

    def _bucket_of(self, token_id: int, pos: int) -> int:
        h = _int_from_hash(self.key_bytes, pos.to_bytes(4, "big"), token_id.to_bytes(4, "big"))
        return h % self.cfg.num_buckets

    def is_green(self, token_id: int, prefix_ids: List[int], pos: int, mask: List[bool]) -> bool:
        return mask[self._bucket_of(token_id, pos)]

    def bias_logits(self, logits: torch.Tensor, prefix_ids: List[int], pos: int):
        mask = self._green_bucket_mask(prefix_ids, pos)
        for tok in range(logits.shape[-1]):
            if self.is_green(tok, prefix_ids, pos, mask):
                logits[tok] += self.cfg.delta

    def count_hits(self, all_ids: List[int], prompt_ids: List[int]) -> Tuple[int, int]:
        n = len(all_ids) - len(prompt_ids)
        C = 0
        for i in range(len(prompt_ids), len(all_ids)):
            pos = i
            prefix = all_ids[:i]
            token_id = all_ids[i]
            mask = self._green_bucket_mask(prefix, pos)
            if self.is_green(token_id, prefix, pos, mask):
                C += 1
        return C, n

    def z_test(self, C: int, n: int) -> Tuple[float, float]:
        gamma = self.cfg.gamma
        if n == 0 or gamma <= 0 or gamma >= 1:
            return 0.0, 1.0
        mean = n * gamma
        var = n * gamma * (1 - gamma)
        if var <= 0:
            return 0.0, 1.0
        z = (C - mean) / math.sqrt(var)
        p = 1 - norm.cdf(z)
        return z, p
