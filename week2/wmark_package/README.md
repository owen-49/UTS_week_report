# wmark

Minimal LLM watermarking toolkit (soft greenlist + hash buckets). Research/teaching oriented.

## Install

```bash
pip install -e .
# or:
python -m build && pip install dist/wmark-*.whl
```

## Quick Start (Python)

```python
from wmark import WatermarkRunner, GenCfg, SoftWatermark, SoftWMConfig

runner = WatermarkRunner("distilgpt2")
wm = SoftWatermark(runner.tok, SoftWMConfig(gamma=0.5, delta=2.0, key="my-secret"))

text, meta = runner.generate("Write a short paragraph about koalas:", GenCfg(max_new_tokens=120), wm=wm)
res = runner.detect(wm, meta["token_ids"], meta["prompt_ids"])
print(res)
```

## CLI

```bash
# generate with watermark
wm-generate --prompt "Write about koalas" --method soft --key my-secret > out.txt

# detect (needs token_ids & prompt_ids meta json)
wm-detect --method soft --key my-secret --meta meta.json
```
