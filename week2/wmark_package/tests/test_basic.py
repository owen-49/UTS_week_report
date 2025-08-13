from wmark import WatermarkRunner, GenCfg, SoftWatermark, SoftWMConfig

def test_soft_pipeline_smoke():
    runner = WatermarkRunner("distilgpt2")
    wm = SoftWatermark(runner.tok, SoftWMConfig(gamma=0.5, delta=2.0, key="test-key"))
    text, meta = runner.generate("Hello from test:", GenCfg(max_new_tokens=32), wm=wm)
    res = runner.detect(wm, meta["token_ids"], meta["prompt_ids"])
    assert "z" in res and "p_one_sided" in res
