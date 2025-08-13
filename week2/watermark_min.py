
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
    # Pack each int into 4 bytes (big-endian) to avoid ValueError from bytes(list)
    return b"".join(int(x).to_bytes(4, "big", signed=False) for x in ids)

def _rng_from_context(key: bytes, prefix_ids: List[int], pos: int) -> random.Random:
    # FIX: use 4-byte packing for each token id instead of bytes(prefix_ids)
    ctx_hash = _int_from_hash(key, _bytes_from_ids(prefix_ids), pos.to_bytes(4, "big"))
    return random.Random(ctx_hash)

# --------------------------
# 1) SoftWatermark
# --------------------------
@dataclass
class SoftWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    max_green_cap: Optional[int] = None

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
# 2) HashBucketWatermark
# --------------------------
@dataclass
class HBWMConfig:
    gamma: float = 0.5
    delta: float = 2.0
    key: str = "secret-key"
    num_buckets: int = 1024

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
        h = _int_from_hash(self.key_bytes, pos.to_bytes(4, "big"), int(token_id).to_bytes(4, "big", signed=False))
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

# --------------------------
# Runner
# --------------------------
@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

class WatermarkRunner:
    def __init__(self, model_name="distilgpt2", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, gen_cfg: GenCfg = GenCfg(), wm=None):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids = ids["input_ids"]
        all_ids = input_ids[0].tolist()

        for _ in range(gen_cfg.max_new_tokens):
            out = self.model(input_ids=input_ids)
            logits = out.logits[:, -1, :].squeeze(0)

            if wm is not None:
                wm.bias_logits(logits, prefix_ids=all_ids, pos=len(all_ids))

            logits = logits / max(1e-6, gen_cfg.temperature)
            probs = torch.softmax(logits, dim=-1)

            if gen_cfg.top_k and gen_cfg.top_k > 0:
                topk = torch.topk(probs, gen_cfg.top_k)
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[topk.indices] = True
                probs = torch.where(mask, probs, torch.tensor(0., device=probs.device))
                probs = probs / probs.sum()

            if gen_cfg.top_p and 0 < gen_cfg.top_p < 1.0:
                sort_probs, sort_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sort_probs, dim=-1)
                keep = cum <= gen_cfg.top_p
                keep[0] = True
                filtered = torch.zeros_like(probs)
                filtered[sort_idx[keep]] = probs[sort_idx[keep]]
                probs = filtered / filtered.sum()

            next_id = torch.multinomial(probs, num_samples=1).item()
            all_ids.append(next_id)
            input_ids = torch.tensor([all_ids], device=self.device)
            if self.tok.eos_token_id is not None and next_id == self.tok.eos_token_id:
                break

        text = self.tok.decode(all_ids, skip_special_tokens=True)
        meta = {"token_ids": all_ids, "prompt_ids": ids["input_ids"][0].tolist()}
        return text, meta

# --------------------------
# Convenience CLI
# --------------------------
def main_generate():
    p = argparse.ArgumentParser("generate")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["none","soft","hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--meta-out", type=str, default="meta.json")
    args = p.parse_args()

    runner = WatermarkRunner(model_name=args.model)

    wm = None
    if args.method == "soft":
        wm = SoftWatermark(runner.tok, SoftWMConfig(args.gamma, args.delta, args.key, args.max_green_cap))
    elif args.method == "hash":
        wm = HashBucketWatermark(runner.tok, HBWMConfig(args.gamma, args.delta, args.key, args.num_buckets))

    text, meta = runner.generate(
        args.prompt,
        GenCfg(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
        wm=wm
    )
    print(text)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[Saved meta to {args.meta_out}]")

def main_detect():
    p = argparse.ArgumentParser("detect")
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["soft","hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--meta", required=True)
    args = p.parse_args()

    from pathlib import Path
    meta = json.loads(Path(args.meta).read_text())

    runner = WatermarkRunner(model_name=args.model)

    if args.method == "soft":
        wm = SoftWatermark(runner.tok, SoftWMConfig(args.gamma, args.delta, args.key, args.max_green_cap))
    else:
        wm = HashBucketWatermark(runner.tok, HBWMConfig(args.gamma, args.delta, args.key, args.num_buckets))

    C, n = wm.count_hits(meta["token_ids"], meta["prompt_ids"])
    z, pval = wm.z_test(C, n)
    out = {"hits": C, "n": n, "gamma": wm.cfg.gamma, "z": z, "p_one_sided": pval}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        sys.argv.pop(1)
        main_generate()
    elif len(sys.argv) > 1 and sys.argv[1] == "detect":
        sys.argv.pop(1)
        main_detect()
    else:
        print("Usage:\n  python watermark_min.py generate --prompt '...' [options]\n  python watermark_min.py detect --meta meta.json [options]")
