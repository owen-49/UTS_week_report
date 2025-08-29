# LLM_watermark.py
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from synthid_text import SynthIDText, WatermarkConfig, WatermarkDetector


# --------------------------
# 小工具
# --------------------------
def resolve_device(name: str = "auto") -> str:
    """优先 CUDA，否则 CPU（默认不走 MPS 以避免 macOS 崩溃）"""
    n = (name or "auto").lower()
    if n == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return n


@torch.inference_mode()
def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """温度 / top-k / top-p 过滤，返回筛后的 logits（不归一化）"""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    # top-k
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, k)
        keeped = torch.full_like(logits, float("-inf"))
        keeped.scatter_(0, topk_idx, topk_vals)
        logits = keeped

    # top-p
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)

        to_remove = cdf > top_p
        to_remove[1:] = to_remove[:-1].clone()
        to_remove[0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(0, sorted_indices, to_remove)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


# --------------------------
# 配置
# --------------------------
@dataclass
class WatermarkedGenerationConfig:
    """水印生成配置"""
    # 生成参数
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # 水印参数
    watermark_key: str = "default_watermark_key"
    apply_watermark: bool = True
    watermark_config: Optional[WatermarkConfig] = None

    # 其他
    pad_token_id: Optional[int] = None


# --------------------------
# 主类
# --------------------------
class WatermarkedLLM:
    """
    集成 SynthID-Text 水印的轻量化 LLM 包装器
    - 不使用 device_map（无需 accelerate）
    - 默认避开 MPS；单设备 .to(device)
    - 生成循环使用 KV 缓存（use_cache=True）
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.device = resolve_device(device)
        self.model_name = model_name_or_path

        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model from {self.model_name}...")
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device).eval()

        self.watermarker: Optional[SynthIDText] = None
        self.detector: Optional[WatermarkDetector] = None

        print(f"Model loaded successfully on {self.device}")

    # ---- 水印 ----
    def setup_watermark(self, watermark_key: str, config: Optional[WatermarkConfig] = None):
        if config is None:
            config = WatermarkConfig(
                num_layers=12, context_window=4, g_value_dist="bernoulli", repeated_context_masking=True
            )
        self.watermarker = SynthIDText(watermark_key, config)
        self.detector = WatermarkDetector(watermark_key, config)
        print(f"Watermark setup complete (m={config.num_layers}, H={config.context_window})")

    # ---- 生成 ----
    @torch.inference_mode()
    def generate_with_watermark(
        self, prompt: str, config: Optional[WatermarkedGenerationConfig] = None
    ) -> Dict[str, Any]:
        if config is None:
            config = WatermarkedGenerationConfig()

        # 初始化水印器
        if config.apply_watermark and self.watermarker is None:
            self.setup_watermark(config.watermark_key, config.watermark_config)

        # 编码输入
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn_mask = enc.attention_mask.to(self.device)
        prompt_ids = input_ids[0].tolist()
        generated_ids: List[int] = []

        # pad/eos
        pad_token_id = (
            config.pad_token_id or self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        # 重置水印上下文
        if config.apply_watermark and self.watermarker is not None:
            self.watermarker.reset_context_history()

        print(f"Starting generation... (max_new_tokens={config.max_new_tokens}, device={self.device})")
        t0 = time.time()

        # KV 缓存
        past_kv = None
        cur_ids = input_ids  # 首步喂完整前缀

        for _ in range(config.max_new_tokens):
            out = self.model(
                input_ids=cur_ids,
                attention_mask=attn_mask if past_kv is None else None,
                use_cache=True,
                past_key_values=past_kv,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1, :].detach()

            # 解码后处理（温度 / top-k / top-p）
            logits = apply_sampling_filters(
                logits, temperature=config.temperature, top_k=config.top_k, top_p=config.top_p
            )

            # 选择下一个 token
            if config.apply_watermark and self.watermarker is not None and config.do_sample:
                context_tokens = prompt_ids + generated_ids  # 完整上下文
                next_id = self.watermarker.tournament_sampling(logits.cpu(), context_tokens)
            else:
                if config.do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(torch.argmax(logits).item())

            generated_ids.append(next_id)

            # 结束条件
            if self.tokenizer.eos_token_id is not None and next_id == int(self.tokenizer.eos_token_id):
                break

            # 下步仅喂新 token
            cur_ids = torch.tensor([[next_id]], device=self.device)
            attn_mask = None

        dt = time.time() - t0
        full_ids = prompt_ids + generated_ids
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tps = (len(generated_ids) / dt) if dt > 0 else 0.0

        print(f"Generation complete: {len(generated_ids)} tokens in {dt:.2f}s ({tps:.1f} tok/s)")
        return {
            "full_text": full_text,
            "generated_text": gen_text,
            "input_text": prompt,
            "generated_tokens": generated_ids,
            "full_tokens": full_ids,
            "generation_time": dt,
            "tokens_per_second": tps,
            "watermarked": bool(config.apply_watermark),
            "config": config,
            "pad_token_id": pad_token_id,
        }

    # ---- 检测 ----
    def detect_watermark(self, text: str, significance_level: float = 0.01) -> Dict[str, Any]:
        if self.detector is None:
            raise ValueError("Detector not initialized. Call setup_watermark() first.")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        result = self.detector.detect(tokens, significance_level=significance_level)
        result["text"] = text
        result["token_count"] = len(tokens)
        return result

    # ---- 批量 ----
    def batch_generate(
        self, prompts: List[str], config: Optional[WatermarkedGenerationConfig] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for p in prompts:
            if config and config.apply_watermark and self.watermarker:
                self.watermarker.reset_context_history()
            results.append(self.generate_with_watermark(p, config))
        return results


# --------------------------
# 轻量模型推荐配置（可选）
# --------------------------
def get_lightweight_model_configs():
    return {
        "gpt2": {
            "model_name": "gpt2",
            "torch_dtype": torch.float32,
            "watermark_config": WatermarkConfig(num_layers=12, context_window=4),
        },
        "distilgpt2": {
            "model_name": "distilgpt2",
            "torch_dtype": torch.float32,
            "watermark_config": WatermarkConfig(num_layers=10, context_window=3),
        },
        "microsoft/DialoGPT-small": {
            "model_name": "microsoft/DialoGPT-small",
            "torch_dtype": torch.float32,
            "watermark_config": WatermarkConfig(num_layers=10, context_window=4),
        },
    }


# --------------------------
# 演示主程序
# --------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    model_name = "gpt2"  # 可换成 distilgpt2 等
    print("=" * 60)
    print("SynthID-Text 水印集成测试")
    print("=" * 60)

    try:
        llm = WatermarkedLLM(
            model_name_or_path=model_name,
            device="cpu",               # 避开 macOS MPS；有 NVIDIA 时可改为 "cuda"
            torch_dtype=torch.float32,  # 小模型 fp32 即可
        )

        # 配置水印
        wm_key = "test_watermark_key_12345"
        wm_cfg = WatermarkConfig(
            num_layers=12, context_window=4, g_value_dist="bernoulli", repeated_context_masking=True
        )
        llm.setup_watermark(wm_key, wm_cfg)

        prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Climate change is one of the most important challenges",
        ]

        print("\n1) 带水印生成")
        print("-" * 40)
        gen_cfg_wm = WatermarkedGenerationConfig(
            max_new_tokens=50, temperature=0.8, top_p=0.9, top_k=50, apply_watermark=True,
            watermark_key=wm_key, watermark_config=wm_cfg
        )
        wm_results = []
        for i, p in enumerate(prompts, 1):
            print(f"\nPrompt {i}: {p}")
            r = llm.generate_with_watermark(p, gen_cfg_wm)
            wm_results.append(r)
            print(f"Generated: {r['generated_text'][:120]}...")

        print("\n2) 无水印生成")
        print("-" * 40)
        gen_cfg_un = WatermarkedGenerationConfig(
            max_new_tokens=50, temperature=0.8, top_p=0.9, top_k=50, apply_watermark=False
        )
        un_results = []
        for i, p in enumerate(prompts, 1):
            print(f"\nPrompt {i}: {p}")
            r = llm.generate_with_watermark(p, gen_cfg_un)
            un_results.append(r)
            print(f"Generated: {r['generated_text'][:120]}...")

        print("\n3) 水印检测")
        print("-" * 40)
        print("\n带水印文本检测：")
        for i, r in enumerate(wm_results, 1):
            det = llm.detect_watermark(r["generated_text"])
            print(f"Text {i}: WM={det['is_watermarked']}, score={det['score']:.3f}, p={det['p_value']:.6f}")

        print("\n无水印文本检测：")
        for i, r in enumerate(un_results, 1):
            det = llm.detect_watermark(r["generated_text"])
            print(f"Text {i}: WM={det['is_watermarked']}, score={det['score']:.3f}, p={det['p_value']:.6f}")

        print("\n4) 批量生成（带水印）")
        print("-" * 40)
        batch_prompts = [
            "Technology has revolutionized",
            "The importance of education",
            "Environmental protection requires",
        ]
        batch_results = llm.batch_generate(batch_prompts, gen_cfg_wm)
        print(f"Generated {len(batch_results)} texts with watermarks")

        for i, r in enumerate(batch_results, 1):
            det = llm.detect_watermark(r["generated_text"])
            print(f"Batch {i}: WM={det['is_watermarked']}, score={det['score']:.3f}")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
