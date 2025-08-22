# demo.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from synthid import SynthIDConfig, SynthIDWatermarker

model_name = "gpt2"  # 示例：可换为本地可用的 CausalLM
tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
eos_id = tok.eos_token_id

cfg = SynthIDConfig(
    key=12345678901234567890,  # 你的水印密钥（整数）
    H=4,
    m_layers=30,
    N_per_match=2,             # =2 非失真；>2 可失真
    g_kind="bernoulli",
    g_p=0.5,
    nsec_bytes=16,
    temperature=0.7,
    top_k=100,
    top_p=None,
    eos_token_id=eos_id,
    max_new_tokens=200,
)

wm = SynthIDWatermarker(mdl, tok, cfg)

prompt = "Explain why the sky is blue in simple terms:\n"
text_wm = wm.generate(prompt, watermark=True)
print("Watermarked output:\n", text_wm)

# 检测
det = wm.detect(text_wm)
print("\nDetection:", det)

# 对照：无水印生成 + 检测
text_plain = wm.generate(prompt, watermark=False)
print("\nPlain output:\n", text_plain)
det_plain = wm.detect(text_plain)
print("\nDetection (plain):", det_plain)
