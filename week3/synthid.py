import math
import hmac
import hashlib
import struct
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import torch
import torch.nn.functional as F

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------------------
# 低层工具：top-k / top-p / 温度 后处理
# ---------------------------
def apply_temperature_topk_topp(logits: torch.Tensor,
                                temperature: float = 1.0,
                                top_k: Optional[int] = None,
                                top_p: Optional[float] = None) -> torch.Tensor:
    # 温度
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    # top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        thresh = v[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)

    # top-p
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        mask = cdf > top_p
        # 保留第一个超过 top_p 的位置
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
        # 还原原索引位置
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


# ---------------------------
# PRF / Hash：实现论文中的伪随机函数族
# ---------------------------
def int_to_bytes(n: int, length: int = 16) -> bytes:
    return n.to_bytes(length, byteorder='big', signed=False)

def hmac_digest(key_bytes: bytes, msg: bytes, digest_len: int = 16) -> bytes:
    # HMAC-SHA256，取前 digest_len 字节
    return hmac.new(key_bytes, msg, hashlib.sha256).digest()[:digest_len]

def digest_to_uint(d: bytes) -> int:
    return int.from_bytes(d, byteorder='big', signed=False)

# ---------------------------
# 滑动窗随机种子 r_t = h(x_{t-H..t-1}, k)
# ---------------------------
def sliding_window_seed(prev_tokens: List[int], key: int, H: int = 4, nsec_bytes: int = 16) -> int:
    """
    prev_tokens: 生成到 t-1 的 token 列表（IDs）
    key: 水印密钥（整数）
    """
    window = prev_tokens[-H:] if len(prev_tokens) >= H else prev_tokens
    key_b = int_to_bytes(key, nsec_bytes)
    # msg = token ids (8 bytes each) | H
    msg = b''.join(struct.pack('>Q', int(tok)) for tok in window) + struct.pack('>I', H)
    dig = hmac_digest(key_b, msg, digest_len=nsec_bytes)
    return digest_to_uint(dig)

# ---------------------------
# g 值：g_ell(x, r)  ~ F_g  (论文定义4；这里提供 Bernoulli(0.5) 和 Uniform[0,1])
# ---------------------------
@dataclass
class GDist:
    kind: str = "bernoulli"  # "bernoulli" or "uniform"
    p: float = 0.5           # 对 bernoulli 有效
    nsec_bytes: int = 16     # 安全参数字节数（128bit）

    def g_value(self, token_id: int, ell: int, r_seed: int) -> float:
        # 伪随机函数族 h_r(x, ell)：用 r_seed 作为 HMAC key
        key_b = int_to_bytes(r_seed, self.nsec_bytes)
        msg = struct.pack('>QQ', int(token_id), int(ell))
        dig = hmac_digest(key_b, msg, digest_len=self.nsec_bytes)
        u = digest_to_uint(dig) / float(1 << (8 * self.nsec_bytes))  # 近似 U[0,1)

        if self.kind == "bernoulli":
            return 1.0 if u >= (1.0 - self.p) else 0.0  # p=0.5 => 阶跃
        elif self.kind == "uniform":
            return u
        else:
            raise ValueError("Unknown g distribution")

# ---------------------------
# 多层 Tournament sampling（算法2）
# ---------------------------
def tournament_sample(probs: torch.Tensor,
                      r_seed: int,
                      m_layers: int = 30,
                      N_candidates_per_match: int = 2,
                      gdist: Optional[GDist] = None) -> int:
    """
    probs: [V] 归一化后的概率（已应用温度/TopK/TopP）
    返回：选中的 token id（int）
    """
    assert probs.dim() == 1
    V = probs.size(0)
    if gdist is None:
        gdist = GDist(kind="bernoulli", p=0.5)

    # 预采样 M = N^m 个候选，允许重复（论文：第一步过生成 N^m 个样本）
    M = int(N_candidates_per_match ** m_layers)
    # 从概率分布中采样 M 个 token id
    candidates = torch.multinomial(probs, num_samples=M, replacement=True).tolist()

    # 分层锦标赛
    current = candidates
    for ell in range(1, m_layers + 1):
        next_round = []
        # 按 N 个一组
        for j in range(0, len(current), N_candidates_per_match):
            group = current[j:j + N_candidates_per_match]
            # 计算 g 值，选择最大者；平局随机
            g_vals = [gdist.g_value(tok, ell, r_seed) for tok in group]
            max_g = max(g_vals)
            winners = [tok for tok, gv in zip(group, g_vals) if gv == max_g]
            chosen = random.choice(winners)
            next_round.append(chosen)
        current = next_round

    assert len(current) == 1
    return int(current[0])

# ---------------------------
# 重复上下文屏蔽（算法3 K-sequence；此处实现 K=1，亦可扩展）
# ---------------------------
@dataclass
class SynthIDConfig:
    key: int
    H: int = 4
    m_layers: int = 30
    N_per_match: int = 2     # =2 非失真；>2 可失真
    g_kind: str = "bernoulli"
    g_p: float = 0.5
    nsec_bytes: int = 16
    temperature: float = 1.0
    top_k: Optional[int] = 100
    top_p: Optional[float] = None
    disable_masking: bool = False  # 置 True 可关闭重复上下文屏蔽
    max_new_tokens: int = 200
    eos_token_id: Optional[int] = None

class SynthIDWatermarker:
    def __init__(self, model, tokenizer, cfg: SynthIDConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.gdist = GDist(kind=cfg.g_kind, p=cfg.g_p, nsec_bytes=cfg.nsec_bytes)

    @torch.no_grad()
    def generate(self, prompt: str, watermark: bool = True) -> str:
        device = self.model.device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated = input_ids.clone()
        seen_contexts: Set[Tuple[int, ...]] = set()  # K=1：仅当前响应

        for _ in range(self.cfg.max_new_tokens):
            out = self.model(input_ids=generated)
            logits = out.logits[:, -1, :]  # [1, V]
            logits = apply_temperature_topk_topp(
                logits.squeeze(0),
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p
            )
            probs = F.softmax(logits, dim=-1)

            if not watermark:
                # 常规抽样
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # 检查重复上下文屏蔽
                prev = generated[0].tolist()
                ctx = tuple(prev[-self.cfg.H:]) if len(prev) >= self.cfg.H else tuple(prev)
                if (not self.cfg.disable_masking) and (ctx in seen_contexts):
                    # 禁用水印，本步常规采样
                    next_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    # 生成随机种子
                    r_seed = sliding_window_seed(prev, self.cfg.key, H=self.cfg.H, nsec_bytes=self.cfg.nsec_bytes)
                    # Tournament sampling
                    next_id = tournament_sample(
                        probs,
                        r_seed=r_seed,
                        m_layers=self.cfg.m_layers,
                        N_candidates_per_match=self.cfg.N_per_match,
                        gdist=self.gdist
                    )
                    # 记录本次上下文
                    seen_contexts.add(ctx)

            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
            if self.cfg.eos_token_id is not None and next_id == self.cfg.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    # 检测：均值分数 + z 检验（无需访问底模）
    def detect(self, text: str) -> Dict[str, float]:
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        H = self.cfg.H
        m = self.cfg.m_layers
        g_vals_all = []

        # 从第 H+1 个 token 开始（此前无法形成完整窗口）
        for t in range(H, len(ids)):
            prev = ids[:t]
            r_seed = sliding_window_seed(prev, self.cfg.key, H=H, nsec_bytes=self.cfg.nsec_bytes)
            xt = ids[t]
            for ell in range(1, m + 1):
                gv = self.gdist.g_value(xt, ell, r_seed)
                g_vals_all.append(gv)

        if len(g_vals_all) == 0:
            return {"mean_g": float('nan'), "z": float('nan'), "p_value": float('nan'), "n_evidence": 0}

        mean_g = float(sum(g_vals_all) / len(g_vals_all))
        # 频率学：Bernoulli(0.5) 的均值检验
        p0 = self.gdist.p if self.gdist.kind == "bernoulli" else 0.5
        var = p0 * (1 - p0)
        n = len(g_vals_all)
        z = (mean_g - p0) / math.sqrt(var / n + 1e-12)

        if HAS_SCIPY:
            # 双尾 p 值
            p_value = 2 * (1 - norm.cdf(abs(z)))
        else:
            # 正态近似的粗略 p 值（无 SciPy 时）
            # 68-95-99.7 规则的启发式；仅供参考
            if abs(z) < 1.0:
                p_value = 0.32
            elif abs(z) < 2.0:
                p_value = 0.05
            elif abs(z) < 3.0:
                p_value = 0.003
            else:
                p_value = 1e-4

        return {"mean_g": mean_g, "z": z, "p_value": p_value, "n_evidence": n}
