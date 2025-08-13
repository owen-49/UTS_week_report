from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

class WatermarkRunner:
    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        gen_cfg: GenCfg = GenCfg(),
        wm=None,  # SoftWatermark / HashBucketWatermark / None
    ) -> Tuple[str, Dict]:
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

            # top-k
            if gen_cfg.top_k and gen_cfg.top_k > 0:
                topk = torch.topk(probs, gen_cfg.top_k)
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[topk.indices] = True
                probs = torch.where(mask, probs, torch.tensor(0., device=probs.device))
                probs = probs / probs.sum()

            # top-p
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
        return text, {"token_ids": all_ids, "prompt_ids": ids["input_ids"][0].tolist()}

    def detect(self, wm, token_ids: List[int], prompt_ids: List[int]) -> Dict:
        C, n = wm.count_hits(token_ids, prompt_ids)
        z, p = wm.z_test(C, n)
        return {"hits": C, "n": n, "gamma": wm.cfg.gamma, "z": z, "p_one_sided": p}
