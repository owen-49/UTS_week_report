import json
import argparse
from .runner import WatermarkRunner, GenCfg
from .watermark import (
    SoftWatermark, SoftWMConfig,
    HashBucketWatermark, HBWMConfig
)

def _build_wm(args, tokenizer):
    if args.method == "soft":
        return SoftWatermark(tokenizer, SoftWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key, max_green_cap=args.max_green_cap
        ))
    elif args.method == "hash":
        return HashBucketWatermark(tokenizer, HBWMConfig(
            gamma=args.gamma, delta=args.delta, key=args.key, num_buckets=args.num_buckets
        ))
    else:
        return None

def main_generate():
    p = argparse.ArgumentParser("wm-generate", description="Generate text with (or without) watermark")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["none", "soft", "hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    args = p.parse_args()

    runner = WatermarkRunner(model_name=args.model)
    wm = None
    if args.method != "none":
        wm = _build_wm(args, runner.tok)

    text, meta = runner.generate(
        args.prompt,
        GenCfg(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
        wm=wm
    )
    print(text)
    print("\n[METADATA]")
    print(json.dumps(meta, indent=2))

def main_detect():
    p = argparse.ArgumentParser("wm-detect", description="Detect watermark on a tokenized sequence")
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--method", choices=["soft", "hash"], default="soft")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--key", type=str, default="secret-key")
    p.add_argument("--max-green-cap", type=int, default=None)
    p.add_argument("--num-buckets", type=int, default=2048)
    p.add_argument("--meta", required=True, help="Path to json containing token_ids & prompt_ids")
    args = p.parse_args()

    import json, sys, pathlib
    meta_path = pathlib.Path(args.meta)
    if not meta_path.exists():
        print(f"[ERROR] meta json not found: {meta_path}", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(meta_path.read_text())

    runner = WatermarkRunner(model_name=args.model)
    wm = _build_wm(args, runner.tok)

    res = runner.detect(wm, meta["token_ids"], meta["prompt_ids"])
    print(json.dumps(res, indent=2))
