#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, hmac, hashlib, json, math, random, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.stats import binom

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ========== SynthID-Text (精简可评测版) ==========

@dataclass
class WatermarkConfig:
    num_layers: int = 10         # m
    context_window: int = 4      # H
    g_value_dist: str = "bernoulli"   # "bernoulli" or "uniform"
    max_candidates: int = 2048   # cap for pre-sampled candidates (避免 2**m 爆炸)
    repeated_context_masking: bool = True

class SynthIDText:
    def __init__(self, watermark_key: str, cfg: WatermarkConfig):
        self.cfg = cfg
        # key 作为 HMAC 的 key（bytes）
        self.key = watermark_key.encode("utf-8")
        # 记录本次响应中出现过的 H 长度窗口（K-sequence=1 的常见做法）
        self.seen_contexts = set()

    # -------- PRF / 随机种子与 g 值 --------
    def _hmac(self, msg: bytes) -> bytes:
        return hmac.new(self.key, msg, hashlib.sha256).digest()

    def _seed_from_context(self, ctx_tokens: List[int]) -> int:
        window = ctx_tokens[-self.cfg.context_window:] if len(ctx_tokens) >= self.cfg.context_window else ctx_tokens
        # 二进制更稳：每个 token 8 字节 + H 4 字节
        msg = b''.join(int(t).to_bytes(8,'big') for t in window) + int(self.cfg.context_window).to_bytes(4,'big')
        d = self._hmac(msg)
        return int.from_bytes(d[:8], 'big')   # 64-bit seed

    def _g_value(self, token_id: int, r_seed: int, layer: int) -> float:
        msg = int(token_id).to_bytes(8,'big') + int(layer).to_bytes(4,'big') + int(r_seed).to_bytes(8,'big')
        d = self._hmac(msg)
        u = int.from_bytes(d[:8], 'big') / float(1<<64)  # ~U[0,1)
        if self.cfg.g_value_dist == "bernoulli":
            return 1.0 if u >= 0.5 else 0.0
        elif self.cfg.g_value_dist == "uniform":
            return u
        else:
            raise ValueError("Unknown g_value_dist")

    # -------- 锦标赛采样（N=2，预采样 capped） --------
    def _single_layer(self, pair: List[int], r_seed: int, layer: int) -> int:
        if len(pair) == 1:
            return pair[0]
        g1 = self._g_value(pair[0], r_seed, layer)
        g2 = self._g_value(pair[1], r_seed, layer)
        if g1 > g2: return pair[0]
        if g2 > g1: return pair[1]
        return random.choice(pair)  # 平局

    def tournament_sample(self, probs: np.ndarray, prev_tokens: List[int]) -> int:
        # 重复上下文屏蔽（同一响应内，任一已见窗口再次出现则不加水印）
        if self.cfg.repeated_context_masking:
            ctx = tuple(prev_tokens[-self.cfg.context_window:])
            if ctx in self.seen_contexts:
                # 普通采样
                return int(np.random.choice(len(probs), p=probs))
            self.seen_contexts.add(ctx)

        # 生成随机种子
        r_seed = self._seed_from_context(prev_tokens)

        # 预采样 M = min(2^m, cap)
        M_exp = 1 << self.cfg.num_layers  # 2**m
        M = min(M_exp, self.cfg.max_candidates)
        cand = np.random.choice(len(probs), size=M, p=probs, replace=True).tolist()

        # m 层两两对决（若 M<2^m，等价于有效层数为 log2(M)）
        for layer in range(1, self.cfg.num_layers + 1):
            if len(cand) == 1:
                break
            nxt = []
            for i in range(0, len(cand), 2):
                pair = cand[i:i+2]
                winner = self._single_layer(pair, r_seed, layer)
                nxt.append(winner)
            cand = nxt
        return int(cand[0])

    # -------- 检测：精确命中计数 + p 值 --------
    def detect_stats(self, tokens: List[int], prefix: Optional[List[int]] = None) -> Tuple[int,int,float]:
        """返回 (hits, total, mean_score)"""
        prefix = prefix or []
        full = prefix + tokens
        hits = 0
        total = 0

        for t, tok in enumerate(tokens):
            ctx = full[:len(prefix)+t]
            r_seed = self._seed_from_context(ctx)
            for layer in range(1, self.cfg.num_layers + 1):
                gv = self._g_value(tok, r_seed, layer)
                if self.cfg.g_value_dist == "bernoulli":
                    hits += (1 if gv == 1.0 else 0)
                else:  # uniform -> 使用阈值 0.5 的“命中”计数，也可直接均值检验
                    hits += (1 if gv >= 0.5 else 0)
                total += 1
        mean_score = hits/total if total else 0.0
        return hits, total, mean_score

    def reset(self):
        self.seen_contexts.clear()

# ========== 评测逻辑（模拟 logits） ==========

def simulate_sequence(wm: SynthIDText, vocab_size: int, seq_len: int, watermarked: bool, prefix=None) -> List[int]:
    """用随机 logits 生成一条序列；若 watermarked=False，则普通采样。"""
    prefix = prefix or [1,2,3]
    tokens: List[int] = []
    wm.reset()

    for _ in range(seq_len):
        # 随机 logits -> softmax
        logits = np.random.randn(vocab_size).astype(np.float64) * 2.0
        logits -= logits.max()
        probs = np.exp(logits); probs /= probs.sum()

        if watermarked:
            tok = wm.tournament_sample(probs, prefix + tokens)
        else:
            tok = int(np.random.choice(vocab_size, p=probs))
        tokens.append(tok)
    return tokens

def roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Tuple[np.ndarray,np.ndarray,float]:
    """手写 ROC 和 AUC（避免额外依赖 sklearn）"""
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    # 排序（分数降序）
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]

    # 累积 TP/FP
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    # 去重：每个唯一阈值保留最后一个点
    uniq_idx = np.where(np.diff(scores, append=np.nan) != 0)[0]
    TPR = tps[uniq_idx] / (P if P>0 else 1)
    FPR = fps[uniq_idx] / (N if N>0 else 1)

    # AUC（梯形法则）
    auc = np.trapz(TPR, FPR)
    return FPR, TPR, float(auc)

def quantile_threshold_at_fpr(scores_neg: np.ndarray, target_fpr: float) -> float:
    """负样本分布上选择阈值，使 FPR ≈ target_fpr（右尾分位）"""
    q = 1.0 - target_fpr
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores_neg, q, method="higher"))

def run_eval(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = WatermarkConfig(
        num_layers=args.num_layers,
        context_window=args.context_window,
        g_value_dist=args.g_dist,
        max_candidates=args.max_candidates,
        repeated_context_masking=not args.disable_masking
    )
    wm = SynthIDText(args.key, cfg)

    print(f"[Config] m={cfg.num_layers} H={cfg.context_window} g={cfg.g_value_dist} "
          f"cap={cfg.max_candidates} masking={cfg.repeated_context_masking}")
    print(f"[Data] trials={args.trials} vocab={args.vocab_size} seq_len(s)={args.seq_lens}\n")

    for L in args.seq_lens:
        wm_scores, wm_pvals = [], []
        un_scores, un_pvals = [], []

        # ------- 多次实验 -------
        for _ in range(args.trials):
            # 生成水印文本
            seq_wm = simulate_sequence(wm, args.vocab_size, L, watermarked=True, prefix=[1,2,3])
            hits, n, score = wm.detect_stats(seq_wm, prefix=[1,2,3])
            p = 1.0 - binom.cdf(hits-1, n, 0.5)
            wm_scores.append(score); wm_pvals.append(p)

            # 生成非水印文本
            seq_un = simulate_sequence(wm, args.vocab_size, L, watermarked=False, prefix=[1,2,3])
            hits_u, n_u, score_u = wm.detect_stats(seq_un, prefix=[1,2,3])
            p_u = 1.0 - binom.cdf(hits_u-1, n_u, 0.5)
            un_scores.append(score_u); un_pvals.append(p_u)

        wm_scores = np.asarray(wm_scores); un_scores = np.asarray(un_scores)

        # ------- ROC / AUC -------
        FPR, TPR, auc = roc_auc(wm_scores, un_scores)

        # ------- 阈值（FPR≈1%） -------
        thr_1pct = quantile_threshold_at_fpr(un_scores, 0.01)
        fpr_obs = (un_scores > thr_1pct).mean()
        tpr_obs = (wm_scores > thr_1pct).mean()

        print(f"[Len={L:4d}] "
              f"WM μ={wm_scores.mean():.3f}±{wm_scores.std():.3f} | "
              f"UN μ={un_scores.mean():.3f}±{un_scores.std():.3f} | "
              f"AUC={auc:.4f} | Thr@1%={thr_1pct:.3f} -> FPR={fpr_obs:.3f}, TPR={tpr_obs:.3f}")

        # ------- 导出 -------
        if args.csv_out:
            import csv, os
            header_needed = not os.path.exists(args.csv_out)
            with open(args.csv_out, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow(["seq_len", "wm_score_mean", "wm_score_std",
                                "un_score_mean", "un_score_std",
                                "auc", "thr_fpr1pct", "fpr_obs", "tpr_obs"])
                w.writerow([L, wm_scores.mean(), wm_scores.std(),
                            un_scores.mean(), un_scores.std(),
                            auc, thr_1pct, fpr_obs, tpr_obs])

        if args.plot and HAS_PLT:
            plt.figure()
            plt.plot(FPR, TPR, label=f"ROC (AUC={auc:.3f})")
            plt.plot([0,1],[0,1],"--",label="chance")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC (seq_len={L})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            out_path = args.plot.replace(".png", f"_L{L}.png")
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"    -> ROC saved to {out_path}")

    if args.plot and not HAS_PLT:
        print("[WARN] matplotlib 未安装，无法绘图；可运行：python -m pip install matplotlib")

# ========== CLI ==========

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate SynthID-Text watermarking (simulation).")
    ap.add_argument("--key", type=str, default="my_secret_key_123", help="watermark key")
    ap.add_argument("--num-layers", type=int, default=10, dest="num_layers")
    ap.add_argument("--context-window", type=int, default=4, dest="context_window")
    ap.add_argument("--g-dist", choices=["bernoulli","uniform"], default="bernoulli")
    ap.add_argument("--max-candidates", type=int, default=2048)

    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[20,50,100,200])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--disable-masking", action="store_true", help="disable repeated-context masking")
    ap.add_argument("--csv-out", type=str, default=None, help="append results to CSV")
    ap.add_argument("--plot", type=str, default=None, help="save ROC figure(s) to this path; '_L{len}.png' will be appended")
    return ap.parse_args()

if __name__ == "__main__":
    run_eval(parse_args())
