#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft Watermark + 安全采样 + 标点护栏
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
    return b"".join(int(x).to_bytes(4, "big", signed=False) for x in ids)

def _rng_from_context(key: bytes, prefix_ids: List[int], pos: int) -> random.Random:
    ctx_hash = _int_from_hash(key, _bytes_from_ids(prefix_ids), int(pos).to_bytes(4, "big"))
    return random.Random(ctx_hash)

def _build_punct_id_set(tokenizer):
    ids = set()
    try:
        SPACE_ID = tokenizer.encode(" ")[0]
    except Exception:
        SPACE_ID = None

    CANDIDATE_STRINGS = [
        "!", ".", ",", "?", ";", ":", "…", "—", "–",
        "!!", "!!!", "??", "???", "...",
        " !", " .", " ,", " ?", " ;", " :", " …",
        " !!", " !!!", " ??", " ???", " ..."
    ]

    for s in CANDIDATE_STRINGS:
        try:
            enc = tokenizer.encode(s, add_special_tokens=False)
            for tid in enc:
                if SPACE_ID is not None and tid == SPACE_ID:
                    continue
                ids.add(int(tid))
        except:
            pass
    return ids

# --------------------------
# Soft Watermark
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

    def _green_ids_at(self, prefix_ids: List[int], pos: int, vocab_size: int):
        k = int(max(1, math.floor(self.cfg.gamma * vocab_size)))
        if self.cfg.max_green_cap:
            k = min(k, self.cfg.max_green_cap)
        rng = _rng_from_context(self.key_bytes, prefix_ids, pos)
        return rng.sample(range(vocab_size), k)

    def bias_logits(self, logits, prefix_ids: List[int], pos: int):
        green = self._green_ids_at(prefix_ids, pos, logits.shape[-1])
        V = logits.shape[-1]
        K = min(self.cfg.topk_intersect, V)
        topk_idx = torch.topk(logits, K).indices.tolist()
        selected = [tid for tid in green if tid in topk_idx]
        if selected:
            logits[torch.tensor(selected, device=logits.device)] += float(self.cfg.delta)

    def count_hits(self, all_ids: List[int], prompt_ids: List[int]) -> Tuple[int, int]:
        vocab_size = self.tok.vocab_size
        n = len(all_ids) - len(prompt_ids)
        C = 0
        for i in range(len(prompt_ids), len(all_ids)):
            green = self._green_ids_at(all_ids[:i], i, vocab_size)
            if all_ids[i] in green:
                C += 1
        return C, n

    def z_test(self, C: int, n: int):
        gamma = self.cfg.gamma
        if n == 0:
            return 0.0, 1.0
        mean = n * gamma
        var = n * gamma * (1 - gamma)
        if var <= 0:
            return 0.0, 1.0
        z = (C - mean) / math.sqrt(var)
        return z, 1 - norm.cdf(z)

# --------------------------
# Generation Config
# --------------------------
@dataclass
class GenCfg:
    min_new_tokens: int = 50
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.1
    repetition_window: int = 256
    freq_penalty: float = 0.8
    freq_window: int = 512
    no_repeat_ngram: int = 3

# --------------------------
# Runner
# --------------------------
class WatermarkRunner:
    def __init__(self, model_name="distilgpt2"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.punct_ids = _build_punct_id_set(self.tok)

    @torch.no_grad()
    def generate(self, prompt: str, gen_cfg: GenCfg, wm=None):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        all_ids = ids["input_ids"][0].tolist()

        for step in range(gen_cfg.max_new_tokens):
            logits = self.model(input_ids=torch.tensor([all_ids], device=self.device)).logits[:, -1, :].squeeze(0)

            # Repetition penalty
            if gen_cfg.repetition_penalty > 1.0 and len(all_ids) > 0:
                recent = set(all_ids[-gen_cfg.repetition_window:])
                logits[list(recent)] /= gen_cfg.repetition_penalty

            # Watermark
            if wm:
                wm.bias_logits(logits, all_ids, len(all_ids))

            # Frequency penalty
            if gen_cfg.freq_penalty > 0:
                from collections import Counter
                cnt = Counter(all_ids[-gen_cfg.freq_window:])
                for tid, c in cnt.items():
                    logits[tid] -= gen_cfg.freq_penalty * math.log1p(c)

            # no-repeat ngram
            if gen_cfg.no_repeat_ngram >= 2 and len(all_ids) >= gen_cfg.no_repeat_ngram:
                n = gen_cfg.no_repeat_ngram
                prefix_seq = all_ids[-(n - 1):]
                blocked = {all_ids[i + n - 1] for i in range(len(all_ids) - n + 1)
                           if all_ids[i:i + n - 1] == prefix_seq}
                logits[list(blocked)] = -1e9

            # Softmax
            probs = torch.softmax(logits / max(gen_cfg.temperature, 1e-6), dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = torch.clamp(probs, min=0.0)

            # Top-k
            if gen_cfg.top_k > 0:
                topk_idx = torch.topk(probs, gen_cfg.top_k).indices
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[topk_idx] = True
                probs = torch.where(mask, probs, torch.tensor(0., device=probs.device))

            # Top-p
            if 0 < gen_cfg.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                keep = cum_probs <= gen_cfg.top_p
                keep[0] = True
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[sorted_idx[keep]] = True
                probs = torch.where(mask, probs, torch.tensor(0., device=probs.device))

            # 防止全是标点
            punct_mask = torch.tensor([tid in self.punct_ids for tid in range(probs.size(0))],
                                      device=probs.device)
            if len(all_ids) >= 2:
                if punct_mask[all_ids[-1]] and punct_mask[all_ids[-2]]:
                    nonpunct_sum = probs[~punct_mask].sum()
                    if nonpunct_sum > 0:
                        probs[punct_mask] = 0
                        probs /= probs.sum()

            if probs.sum() <= 0 or not torch.isfinite(probs.sum()):
                next_id = int(torch.argmax(logits))
            else:
                next_id = int(torch.multinomial(probs / probs.sum(), 1))

            all_ids.append(next_id)

            if step >= gen_cfg.min_new_tokens and next_id == self.tok.eos_token_id:
                break

        text = self.tok.decode(all_ids, skip_special_tokens=True)
        meta = {"token_ids": all_ids, "prompt_ids": ids["input_ids"][0].tolist()}
        return text, meta

# --------------------------
# CLI
# --------------------------
def main_generate():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--wm-topk", type=int, default=100)
    p.add_argument("--min-new-tokens", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--repetition-window", type=int, default=256)
    p.add_argument("--freq-penalty", type=float, default=0.8)
    p.add_argument("--freq-window", type=int, default=512)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--meta-out", type=str, default="meta.json")
    args = p.parse_args()

    runner = WatermarkRunner(args.model)
    wm = SoftWatermark(runner.tok, SoftWMConfig(
        gamma=args.gamma, delta=args.delta, key=args.key,
        max_green_cap=args.max_green_cap, topk_intersect=args.wm_topk
    ))

    text, meta = runner.generate(args.prompt, GenCfg(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        freq_penalty=args.freq_penalty,
        freq_window=args.freq_window,
        no_repeat_ngram=args.no_repeat_ngram
    ), wm=wm)

    print(text)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved meta to {args.meta_out}]")

def main_detect():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--meta", required=True)
    p.add_argument("--wm-topk", type=int, default=100)
    args = p.parse_args()

    from pathlib import Path
    meta = json.loads(Path(args.meta).read_text())
    runner = WatermarkRunner(args.model)
    wm = SoftWatermark(runner.tok, SoftWMConfig(
        gamma=args.gamma, delta=args.delta, key=args.key,
        max_green_cap=args.max_green_cap, topk_intersect=args.wm_topk
    ))

    C, n = wm.count_hits(meta["token_ids"], meta["prompt_ids"])
    z, pval = wm.z_test(C, n)
    print(json.dumps({"hits": C, "n": n, "gamma": wm.cfg.gamma, "z": z, "p_one_sided": pval}, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        sys.argv.pop(1); main_generate()
    elif len(sys.argv) > 1 and sys.argv[1] == "detect":
        sys.argv.pop(1); main_detect()
    else:
        print("Usage:\n  python watermark_min.py generate --prompt '...'\n  python watermark_min.py detect --meta meta.json")
