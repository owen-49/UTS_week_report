# cli.py
import argparse, sys, json, os, secrets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from synthid import SynthIDConfig, SynthIDWatermarker

def auto_device(name: str):
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name

def parse_dtype(name: str | None):
    if not name:
        return None
    name = name.lower()
    if name in ("fp16", "float16"): return torch.float16
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    if name in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

def read_text_arg(prompt: str | None, prompt_file: str | None):
    if prompt is not None:
        return prompt
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No input provided. Use --prompt / --prompt-file or pipe from stdin.")

def read_detect_text(text: str | None, file: str | None):
    if text is not None:
        return text
    if file:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No input provided. Use --text / --file or pipe from stdin.")

def resolve_key(key_arg: str | None, save_path: str | None):
    """
    --key 支持十进制或 0x... 十六进制；若传入 'rand' 或不传，则生成 128bit 随机密钥
    """
    if key_arg is None or key_arg.lower() == "rand":
        key = int.from_bytes(secrets.token_bytes(16), "big")
        sys.stderr.write(f"[INFO] Generated random 128-bit key: {key} (hex: 0x{key:x})\n")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(key))
            sys.stderr.write(f"[INFO] Saved key to {save_path}\n")
        return key
    try:
        if key_arg.lower().startswith("0x"):
            return int(key_arg, 16)
        return int(key_arg)
    except Exception:
        raise SystemExit("Invalid --key. Use decimal, 0xHEX, or 'rand'.")

def load_model_and_tokenizer(model_name: str, device: str, dtype):
    tok = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    mdl = mdl.to(device)
    return mdl, tok

def build_cfg(args, tok):
    eos_id = args.eos_id
    if eos_id is None and tok.eos_token_id is not None:
        eos_id = tok.eos_token_id
    return SynthIDConfig(
        key=args.__key_int__,
        H=args.H,
        m_layers=args.m_layers,
        N_per_match=args.N_per_match,
        g_kind=args.g_kind,
        g_p=args.g_p,
        nsec_bytes=args.nsec_bytes,
        temperature=args.temperature,
        top_k=(None if args.top_k is None or args.top_k <= 0 else args.top_k),
        top_p=args.top_p,
        disable_masking=args.disable_masking,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id
    )

def cmd_gen(args):
    device = auto_device(args.device)
    dtype = parse_dtype(args.dtype)
    mdl, tok = load_model_and_tokenizer(args.model, device, dtype)
    args.__key_int__ = resolve_key(args.key, args.save_key)

    cfg = build_cfg(args, tok)
    wm = SynthIDWatermarker(mdl, tok, cfg)

    prompt = read_text_arg(args.prompt, args.prompt_file)
    out_text = wm.generate(prompt, watermark=not args.no_watermark)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)

    if args.print_stats:
        stats = wm.detect(out_text)
        sys.stderr.write("\n[DETECTION-STATS] " + json.dumps(stats, ensure_ascii=False) + "\n")

def cmd_detect(args):
    device = auto_device(args.device)
    dtype = parse_dtype(args.dtype)
    mdl, tok = load_model_and_tokenizer(args.model, device, dtype)
    args.__key_int__ = resolve_key(args.key, args.save_key)

    cfg = build_cfg(args, tok)
    wm = SynthIDWatermarker(mdl, tok, cfg)

    text = read_detect_text(args.text, args.file)
    stats = wm.detect(text)
    sys.stdout.write(json.dumps(stats, ensure_ascii=False, indent=(2 if args.pretty else None)) + "\n")

def add_common_args(p):
    # 模型与解码
    p.add_argument("--model", default="gpt2", help="Hugging Face model id or local path (causal LM).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default=None, help="float32|float16(fp16)|bfloat16(bf16)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--eos-id", type=int, default=None)

    # 水印参数
    p.add_argument("--key", default="rand", help="Watermark key (decimal or 0xHEX). Use 'rand' to generate.")
    p.add_argument("--save-key", default=None, help="If set, save the key here (on generation or detection).")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--m-layers", type=int, default=30)
    p.add_argument("--N-per-match", type=int, default=2)
    p.add_argument("--g-kind", choices=["bernoulli", "uniform"], default="bernoulli")
    p.add_argument("--g-p", type=float, default=0.5)
    p.add_argument("--nsec-bytes", type=int, default=16)
    p.add_argument("--disable-masking", action="store_true", help="Disable repeated-context masking (K-seq).")

def main():
    ap = argparse.ArgumentParser(prog="synthid-cli", description="SynthID-Text watermarking (generation & detection)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # gen
    pg = sp.add_parser("gen", help="Generate text with (or without) watermark via tournament sampling.")
    add_common_args(pg)
    pg.add_argument("--prompt", type=str, default=None, help="Prompt string.")
    pg.add_argument("--prompt-file", type=str, default=None, help="Read prompt from file.")
    pg.add_argument("--out", type=str, default=None, help="Write generated text to file (default: stdout).")
    pg.add_argument("--no-watermark", action="store_true", help="Generate without watermark (control).")
    pg.add_argument("--print-stats", action="store_true", help="After generation, run detector on the output.")
    pg.set_defaults(func=cmd_gen)

    # detect
    pd = sp.add_parser("detect", help="Detect watermark on given text (needs the correct key).")
    add_common_args(pd)
    pd.add_argument("--text", type=str, default=None, help="Text string to detect.")
    pd.add_argument("--file", type=str, default=None, help="Read text to detect from file.")
    pd.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    pd.set_defaults(func=cmd_detect)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
