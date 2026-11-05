"""General-purpose reward baselines (no physics-specific design), used to ask: do they see what our critics see?

* ``viescore`` -- a VIEScore-STYLE judge: a chat VLM rates semantic consistency (edit success, over-editing) and
  perceptual quality (naturalness, artifacts) on 0-10; SC and PQ are the min of their two sub-scores and
  overall = sqrt(SC * PQ), the combination VIEScore describes. The prompts below are written for this repo; they are
  NOT the official VIEScore prompts, so report this baseline as "VIEScore-style" and say which VLM was the judge.
* EditReward (TIGER-Lab) is run by experiments/score_editreward.py in its own environment (it pins
  transformers==4.57.0, this repo needs >=5.5).
"""

from __future__ import annotations

import math
import re

_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")

SC_PROMPT = (
    'You are evaluating an image edit. Image 1 is the original photo, image 2 is the edited result.\n'
    'Editing instruction: "{instruction}"\n'
    "Rate two things from 0 (worst) to 10 (best):\n"
    "1. Editing success: how well the edited image follows the instruction.\n"
    "2. Over-editing: how well everything the instruction did not ask to change was left unchanged "
    "(10 = no unwanted change).\n"
    'Reply with exactly two numbers separated by a comma, for example "7, 9". No other text.'
)

PQ_PROMPT = (
    "You are evaluating the visual quality of one image.\n"
    "Rate two things from 0 (worst) to 10 (best):\n"
    "1. Naturalness: does it look like a natural photograph that is physically plausible and consistent?\n"
    "2. Artifacts: is it free of visible artifacts such as distortions, halos, blur or cut-out edges "
    "(10 = no artifacts)?\n"
    'Reply with exactly two numbers separated by a comma, for example "6, 8". No other text.'
)


def parse_scores(text: str, n: int = 2, lo: float = 0.0, hi: float = 10.0) -> list[float] | None:
    """First ``n`` numbers in [lo, hi] found in the judge's reply, or None if there are fewer."""
    nums = [x for x in (float(v) for v in _NUM.findall(text)) if lo <= x <= hi]
    return nums[:n] if len(nums) >= n else None


SOFT_PROMPTS = {
    "success": ('Image 1 is the original photo, image 2 is an edited result. Editing instruction: "{instruction}"\n'
                "On a scale from 0 (worst) to 9 (best), how well does the edited image follow the instruction? "
                "Reply with a single digit."),
    "over_edit": ('Image 1 is the original photo, image 2 is an edited result. Editing instruction: "{instruction}"\n'
                  "On a scale from 0 (worst) to 9 (best), how well was everything the instruction did not ask to change "
                  "left unchanged (9 = no unwanted change)? Reply with a single digit."),
    "naturalness": ("On a scale from 0 (worst) to 9 (best), does this image look like a natural photograph that is "
                    "physically plausible and consistent? Reply with a single digit."),
    "artifacts": ("On a scale from 0 (worst) to 9 (best), is this image free of visible artifacts such as distortions, "
                  "halos, blur or cut-out edges (9 = no artifacts)? Reply with a single digit."),
}


def viescore_soft(expect, src, edit, instruction: str) -> dict:
    """Same combination as ``viescore`` but each sub-score is the EXPECTED value over the digit-token probabilities
    (``expect(images, prompt) -> float in [0, 1]``). The integer judge says "8, 9" to almost everything, so its scores tie
    constantly; the expectation keeps the small differences the logits carry."""
    out = {k: expect([src, edit] if k in ("success", "over_edit") else [edit], p.format(instruction=instruction))
           for k, p in SOFT_PROMPTS.items()}
    out["sc"], out["pq"] = min(out["success"], out["over_edit"]), min(out["naturalness"], out["artifacts"])
    out["overall"] = math.sqrt(out["sc"] * out["pq"])
    return out


def viescore(judge, src, edit, instruction: str) -> dict:
    """``judge(images, prompt) -> str``. Returns the four sub-scores, SC, PQ and overall, all in [0, 1] (or None)."""
    raw_sc = judge([src, edit], SC_PROMPT.format(instruction=instruction))
    raw_pq = judge([edit], PQ_PROMPT)
    sc2, pq2 = parse_scores(raw_sc), parse_scores(raw_pq)
    out = {"success": None, "over_edit": None, "naturalness": None, "artifacts": None,
           "sc": None, "pq": None, "overall": None, "raw_sc": raw_sc, "raw_pq": raw_pq}
    if sc2:
        out["success"], out["over_edit"] = sc2[0] / 10, sc2[1] / 10
        out["sc"] = min(sc2) / 10
    if pq2:
        out["naturalness"], out["artifacts"] = pq2[0] / 10, pq2[1] / 10
        out["pq"] = min(pq2) / 10
    if out["sc"] is not None and out["pq"] is not None:
        out["overall"] = math.sqrt(out["sc"] * out["pq"])
    return out


class HFChatJudge:
    """Greedy text generation from a chat VLM bundle (src.models.registry.ModelBundle)."""

    def __init__(self, bundle, max_side: int = 768):
        self.b = bundle
        self.max_side = max_side
        tok = bundle.processor.tokenizer
        self.digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

    def expect(self, images, prompt: str) -> float:
        """Expected digit (0-9) under the next-token distribution restricted to the ten digit tokens, divided by 9."""
        import torch

        imgs = []
        for im in images:
            im = im.copy()
            im.thumbnail((self.max_side, self.max_side))
            imgs.append(im)
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        proc, model = self.b.processor, self.b.model
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False)
        inputs = proc(text=[text], images=imgs, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.b.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        p = torch.softmax(logits[self.digit_ids], 0)
        return float((p * torch.arange(10, device=p.device)).sum().item() / 9)

    def __call__(self, images, prompt: str, max_new_tokens: int = 24) -> str:
        import torch

        imgs = []
        for im in images:
            im = im.copy()
            im.thumbnail((self.max_side, self.max_side))
            imgs.append(im)
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        proc, model = self.b.processor, self.b.model
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False)
        inputs = proc(text=[text], images=imgs, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.b.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
