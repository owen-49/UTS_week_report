#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watermark_min.py (punct-allowed + anti-punct-collapse)
- 允许标点进入 greenlist（生成时不再过滤标点）
- 加入“标点风暴”护栏：连续标点时优先从非标点采样
- 保留：Top-K 交集偏置 / 重复惩罚 / 频率惩罚 / no-repeat n-gram / 稳定采样
- CLI: generate / detect
"""

import math, hashlib, random, json, argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import norm


# --------------------------
# Utils
# --------------------------
def _int_from_hash(*parts: bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest(), "big")


def _bytes_from_ids(ids: List[int]) -> bytes:
    """Pack each token id into 4 bytes (big-endian)."""
    return b"".join(int(x).to_bytes(4, "big", signed=False) for x in ids)


def _rng_from_context(key: bytes, prefix_ids: List[int], pos: int) -> random.Random:
    """Deterministic RNG from (key, prefix_ids, pos)."""
    ctx_hash = _int_from_hash(key, _bytes_from_ids(prefix_ids), int(pos).to_bytes(4, "big"))
    return random.Random(ctx_hash)


# --------------------------
# Common helpers
# --------------------------
_PUNCT_TOKENS = {
    "!", "!!", "!!!", ".", "..", "...", ",", ":", ";", "?", "??", "???",
    '"', "'", "``", "''", "(", ")", "[", "]", "{", "}", "-", "—", "–", "…", "’", "“", "”"
}

def _build_punct_id_set(tokenizer):
    """
    构建“标点 token id 集合”，覆盖 GPT-2 中常见的“单标点”“空格+标点”“多标点”形态。
    关键点：排除纯空格 token（否则会把空格当标点，副作用很大）。
    """
    ids = set()

    # 1) 先找出“纯空格” token id（GPT-2 常是 220，但我们动态取）
    try:
        SPACE_ID = tokenizer.encode(" ")[0]
    except Exception:
        SPACE_ID = None

    # 2) 枚举一批常见的标点序列（包含空格前缀与多重标点）
    #    这些会被 tokenizer.encode 分成 1~2 个 token；我们把除了纯空格以外的 id 都加入集合
    CANDIDATE_STRINGS = [
        # 单个标点（无空格）
        "!", ".", ",", "?", ";", ":", "…", "—", "–",
        "!!", "!!!", "??", "???", "...",

        # 前置空格 + 标点（GPT-2 经常把“空格+标点”压成一个 token）
        " !", " .", " ,", " ?", " ;", " :", " …",
        " !!", " !!!", " ??", " ???", " ...",
    ]

    for s in CANDIDATE_STRINGS:
        try:
            enc = tokenizer.encode(s, add_special_tokens=False)
            for tid in enc:
                if SPACE_ID is not None and tid == SPACE_ID:
                    continue  # 排除纯空格
                ids.add(int(tid))
        except Exception:
            pass

    # 3) 兜底：再尝试把几类“裸标点符号”转 id（有些模型支持 convert_tokens_to_ids）
    for s in ["!", ".", ",", "?", ";", ":", "…", "—", "–"]:
        try:
            tid = int(tokenizer.convert_tokens_to_ids(s))
            if SPACE_ID is None or tid != SPACE_ID:
                ids.add(tid)
        except Exception:
            pass

    return ids



def _apply_topk_intersection_bias(
    logits: torch.Tensor,
    candidates: List[int],
    delta: float,
    K: int,
):
    """只对 (候选 ∩ 当前logits Top-K) 加偏置；不再过滤标点。"""
    V = logits.shape[-1]
    K = int(max(1, min(K, V)))
    topk_vals, topk_idx = torch.topk(logits, K)
    topk_set = set(int(i) for i in topk_idx.tolist())
    selected = [tid for tid in candidates if tid in topk_set]
    if selected:
        logits[torch.tensor(selected, device=logits.device, dtype=torch.long)] += float(delta)


# --------------------------
# 1) SoftWatermark
# --------------------------
@dataclass
class SoftWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    max_green_cap: Optional[int] = None
    topk_intersect: int = 100


class SoftWatermark:
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

    def bias_logits(self, logits: torch.Tensor, prefix_ids: List[int], pos: int) -> None:
        green = self._green_ids_at(prefix_ids, pos, logits.shape[-1])
        _apply_topk_intersection_bias(logits, green, self.cfg.delta, self.cfg.topk_intersect)

    # 检测阶段使用“原始定义”的 green set（与生成时一致的 gamma/pos/key），不进行标点或Top-K过滤
    def count_hits(self, all_ids: List[int], prompt_ids: List[int]) -> Tuple[int, int]:
        vocab_size = self.tok.vocab_size
        n = len(all_ids) - len(prompt_ids)
        C = 0
        for i in range(len(prompt_ids), len(all_ids)):
            pos = i
            token_id = int(all_ids[i])
            green = self._green_ids_at(all_ids[:i], pos, vocab_size)
            if token_id in green:
                C += 1
        return C, n

    def z_test(self, C: int, n: int) -> Tuple[float, float]:
        gamma = self.cfg.gamma
        if n == 0 or not (0 < gamma < 1):
            return 0.0, 1.0
        mean = n * gamma
        var = n * gamma * (1 - gamma)
        if var <= 0:
            return 0.0, 1.0
        z = (C - mean) / math.sqrt(var)
        p = 1 - norm.cdf(z)
        return z, p


# --------------------------
# 2) HashBucketWatermark
# --------------------------
@dataclass
class HBWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    num_buckets: int = 1024
    topk_intersect: int = 100


class HashBucketWatermark:
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
        h = _int_from_hash(
            self.key_bytes,
            int(pos).to_bytes(4, "big"),
            int(token_id).to_bytes(4, "big", signed=False),
        )
        return h % self.cfg.num_buckets

    def bias_logits(self, logits: torch.Tensor, prefix_ids: List[int], pos: int) -> None:
        mask = self._green_bucket_mask(prefix_ids, pos)
        V = logits.shape[-1]
        candidates = [tok for tok in range(V) if mask[self._bucket_of(tok, pos)]]
        _apply_topk_intersection_bias(logits, candidates, self.cfg.delta, self.cfg.topk_intersect)

    def count_hits(self, all_ids: List[int], prompt_ids: List[int]) -> Tuple[int, int]:
        n = len(all_ids) - len(prompt_ids)
        C = 0
        for i in range(len(prompt_ids), len(all_ids)):
            pos = i
            token_id = int(all_ids[i])
            mask = self._green_bucket_mask(all_ids[:i], pos)
            if mask[self._bucket_of(token_id, pos)]:
                C += 1
        return C, n

    def z_test(self, C: int, n: int) -> Tuple[float, float]:
        gamma = self.cfg.gamma
        if n == 0 or not (0 < gamma < 1):
            return 0.0, 1.0
        mean = n * gamma
        var = n * gamma * (1 - gamma)
        if var <= 0:
            return 0.0, 1.0
        z = (C - mean) / math.sqrt(var)
        p = 1 - norm.cdf(z)
        return z, p


# --------------------------
# Runner (decoding with penalties, n-gram block, anti-punct-collapse)
# --------------------------
@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.25
    repetition_window: int = 256
    freq_penalty: float = 0.8          # 0~1, larger -> stronger penalty
    freq_window: int = 512
    no_repeat_ngram: int = 3           # 0/1/2 to disable; >=3 typical


class WatermarkRunner:
    def __init__(self, model_name="distilgpt2", device: Optional[str] = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 缓存标点 id 与掩码（供“标点风暴”护栏使用）
        self.punct_ids = _build_punct_id_set(self.tok)
        self._punct_mask_cache = None  # device-bound

    def _punct_mask(self, vocab_size: int):
        if self._punct_mask_cache is None or self._punct_mask_cache.device != torch.device(self.device):
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
            for tid in self.punct_ids:
                if 0 <= tid < vocab_size:
                    mask[tid] = True
            self._punct_mask_cache = mask
        return self._punct_mask_cache

    @torch.no_grad()
    def generate(self, prompt: str, gen_cfg: GenCfg, wm=None) -> Tuple[str, Dict]:
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids = ids["input_ids"]
        all_ids = input_ids[0].tolist()

        for _ in range(gen_cfg.max_new_tokens):
            out = self.model(input_ids=torch.tensor([all_ids], device=self.device))
            logits = out.logits[:, -1, :].squeeze(0)  # [V]

            # 轻度重复惩罚
            if gen_cfg.repetition_penalty and gen_cfg.repetition_penalty > 1.0:
                window = max(1, int(gen_cfg.repetition_window))
                recent = all_ids[-window:]
                if recent:
                    uniq = torch.tensor(list(set(int(t) for t in recent)), device=logits.device, dtype=torch.long)
                    logits[uniq] /= float(gen_cfg.repetition_penalty)

            # 加水印偏置
            if wm is not None:
                wm.bias_logits(logits, prefix_ids=all_ids, pos=len(all_ids))

            # 频率惩罚（log-count）
            if gen_cfg.freq_penalty and gen_cfg.freq_penalty > 0:
                from collections import Counter
                cnt = Counter(all_ids[-int(gen_cfg.freq_window):])
                if len(cnt) > 0:
                    idx = torch.tensor(list(cnt.keys()), device=logits.device, dtype=torch.long)
                    counts = torch.tensor([cnt[i.item()] for i in idx], device=logits.device, dtype=logits.dtype)
                    logits[idx] = logits[idx] - float(gen_cfg.freq_penalty) * torch.log1p(counts)

            # no-repeat n-gram
            if gen_cfg.no_repeat_ngram and gen_cfg.no_repeat_ngram >= 3 and len(all_ids) >= gen_cfg.no_repeat_ngram - 1:
                n = int(gen_cfg.no_repeat_ngram)
                prefix_seq = all_ids[-(n - 1):]
                blocked = set()
                for i in range(len(all_ids) - n + 1):
                    if all_ids[i:i + n - 1] == prefix_seq:
                        blocked.add(all_ids[i + n - 1])
                if blocked:
                    bi = torch.tensor(list(blocked), device=logits.device, dtype=torch.long)
                    logits[bi] = -1e9

            # 温度 & 稳定 softmax
            t = max(1e-6, float(gen_cfg.temperature))
            logits = logits / t
            logits = logits - torch.max(logits)
            probs = torch.softmax(logits, dim=-1)

            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = torch.clamp(probs, min=0.0)
            s = probs.sum()
            if not torch.isfinite(s) or s <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = probs / s

                # top-k
                if gen_cfg.top_k and gen_cfg.top_k > 0:
                    k = int(gen_cfg.top_k)
                    topk = torch.topk(probs, k)
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask[topk.indices] = True
                    probs = torch.where(mask, probs, torch.tensor(0., device=probs.device))
                    s = probs.sum()
                    if s <= 0 or not torch.isfinite(s):
                        probs = torch.zeros_like(probs)
                        probs[topk.indices[0]] = 1.0
                    else:
                        probs = probs / s

                # top-p
                if gen_cfg.top_p and 0 < gen_cfg.top_p < 1.0:
                    sort_probs, sort_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sort_probs, dim=-1)
                    keep = cum <= gen_cfg.top_p
                    if keep.shape[0] > 0:
                        keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[sort_idx[keep]] = probs[sort_idx[keep]]
                    s = filtered.sum()
                    if s <= 0 or not torch.isfinite(s):
                        filtered = torch.zeros_like(probs)
                        filtered[sort_idx[0]] = 1.0
                    else:
                        filtered = filtered / s
                    probs = filtered

                # —— 护栏：连续标点时优先从“非标点”采样 ——
                # —— 护栏：连续标点时优先从“非标点”采样 —— 
                    punct_mask = self._punct_mask(probs.shape[-1])   # bool 向量：True 表示该 token 是标点（含空格+标点形态）

                    # 最近是否“连续标点”：
                    # 1) 最近两个 token 都在 punct_ids
                    # 2) 或者形态是 [空格, 标点] / [空格+标点, 标点] 等复合情况，我们用 punct_mask 直接判断最后两个是否都是“标点类”
                    last_two_punct = False
                    if len(all_ids) >= 2:
                        a, b = all_ids[-1], all_ids[-2]
                        # 使用 punct_mask 直接判断，而不是只靠“裸标点表”
                        last_two_punct = (
                            (0 <= a < punct_mask.numel() and punct_mask[a].item()) and
                            (0 <= b < punct_mask.numel() and punct_mask[b].item())
                        )

                    nonpunct_sum = (probs[~punct_mask]).sum()

                    if last_two_punct and nonpunct_sum > 0:
                        # 只在非标点上采样（把标点概率清零后重归一化）
                        filtered = torch.zeros_like(probs)
                        filtered[~punct_mask] = probs[~punct_mask]
                        s2 = filtered.sum()
                        if s2 > 0 and torch.isfinite(s2):
                            probs = filtered / s2
                    elif nonpunct_sum == 0:
                        # 极端情况：top-k/top-p 之后全是标点；退回到 logits，在“非标点子集”里取最大者
                        nonpunct_idx = (~punct_mask).nonzero(as_tuple=False).flatten()
                        if nonpunct_idx.numel() > 0:
                            sub_logits = logits[nonpunct_idx]
                            best_i = int(torch.argmax(sub_logits).item())
                            next_id = int(nonpunct_idx[best_i].item())
                            all_ids.append(next_id)
                            input_ids = torch.tensor([all_ids], device=self.device)
                            if self.tok.eos_token_id is not None and next_id == self.tok.eos_token_id:
                                break
                            continue  # 直接进入下一轮


                # 最终采样
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = torch.clamp(probs, min=0.0)
                s = probs.sum()
                if s <= 0 or not torch.isfinite(s):
                    next_id = int(torch.argmax(logits).item())
                else:
                    probs = probs / s
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            all_ids.append(next_id)
            input_ids = torch.tensor([all_ids], device=self.device)
            if self.tok.eos_token_id is not None and next_id == self.tok.eos_token_id:
                break

        text = self.tok.decode(all_ids, skip_special_tokens=True)
        meta = {"token_ids": all_ids, "prompt_ids": ids["input_ids"][0].tolist()}
        return text, meta


# --------------------------
# CLI
# --------------------------
def main_generate():
    p = argparse.ArgumentParser("generate")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["none", "soft", "hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--meta-out", type=str, default="meta.json")
    # watermark shaping
    p.add_argument("--wm-topk", type=int, default=100)
    # penalties
    p.add_argument("--repetition-penalty", type=float, default=1.25)
    p.add_argument("--repetition-window", type=int, default=256)
    p.add_argument("--freq-penalty", type=float, default=0.8)
    p.add_argument("--freq-window", type=int, default=512)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    args = p.parse_args()

    runner = WatermarkRunner(model_name=args.model)
    wm = None
    if args.method == "soft":
        wm = SoftWatermark(runner.tok, SoftWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key,
            max_green_cap=args.max_green_cap, topk_intersect=args.wm_topk
        ))
    elif args.method == "hash":
        wm = HashBucketWatermark(runner.tok, HBWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key,
            num_buckets=args.num_buckets, topk_intersect=args.wm_topk
        ))

    text, meta = runner.generate(
        args.prompt,
        GenCfg(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
               top_p=args.top_p, top_k=args.top_k,
               repetition_penalty=args.repetition_penalty, repetition_window=args.repetition_window,
               freq_penalty=args.freq_penalty, freq_window=args.freq_window,
               no_repeat_ngram=args.no_repeat_ngram),
        wm=wm
    )
    print(text)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved meta to {args.meta_out}]")


def main_detect():
    p = argparse.ArgumentParser("detect")
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["soft", "hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--meta", required=True)
    p.add_argument("--wm-topk", type=int, default=100)  # symmetry placeholder
    args = p.parse_args()

    from pathlib import Path
    meta = json.loads(Path(args.meta).read_text())
    runner = WatermarkRunner(model_name=args.model)

    if args.method == "soft":
        wm = SoftWatermark(runner.tok, SoftWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key,
            max_green_cap=args.max_green_cap, topk_intersect=args.wm_topk
        ))
    else:
        wm = HashBucketWatermark(runner.tok, HBWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key,
            num_buckets=args.num_buckets, topk_intersect=args.wm_topk
        ))

    C, n = wm.count_hits(meta["token_ids"], meta["prompt_ids"])
    z, pval = wm.z_test(C, n)
    out = {"hits": C, "n": n, "gamma": wm.cfg.gamma, "z": z, "p_one_sided": pval}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        sys.argv.pop(1); main_generate()
    elif len(sys.argv) > 1 and sys.argv[1] == "detect":
        sys.argv.pop(1); main_detect()
    else:
        print("Usage:\n  python watermark_min.py generate --prompt '...' [options]\n"
              "  python watermark_min.py detect --meta meta.json [options]")
