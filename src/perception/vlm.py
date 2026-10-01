"""VLM region question answering (innovation B, VLM half).

Each question is bound to a REGION derived from the subject mask (no manual annotation, unlike PICAEval):
the model sees the full image with that region boxed in red plus a zoomed crop of it, and we read the
probability of "Yes" from the next-token logits instead of asking for a 1-10 score. Questions are phrased
so that "Yes" = physically / semantically fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from src.spec import EditSpec
from src.utils.geometry import mask_box


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    region: str  # "full" | "contact" | "subject"


def build_questions(spec: EditSpec) -> list[Question]:
    s, pose = spec.subject, spec.pose
    return [
        Question("support", f"Look at the area inside the red box. Is the {s} physically resting on a solid surface "
                            f"there, rather than floating above it or sinking into it?", "contact"),
        Question("shadow", f"Look at the area inside the red box. Is there a natural shadow or darker contact area "
                           f"on the ground right where the {s} touches it?", "contact"),
        Question("surface", f"Is the {s} {pose} on a surface that makes physical sense in this scene "
                            f"(not on an impossible or melted object)?", "contact"),
        Question("integration", f"Does the {s} look like it naturally belongs in this photo, with no strange objects, "
                                f"cut-out edges, glow or outline wrapped around its body?", "subject"),
        Question("lighting", f"Is the lighting on the {s} (direction, brightness and color) consistent with the "
                             f"lighting of the rest of the scene?", "full"),
        Question("background", f"Is the background of this photo a {spec.target_bg.name}?", "full"),
    ]


def region_box(mask: np.ndarray, region: str, pad: float = 0.15) -> tuple[int, int, int, int] | None:
    """Pixel box of a question's region, derived from the subject mask."""
    box = mask_box(mask)
    h, w = mask.shape
    if box is None or region == "full":
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if region == "contact":
        y1 = y2 - 0.25 * bh
        y2 = y2 + 0.15 * bh
    px, py = pad * bw, pad * (y2 - y1)
    return (int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py)))


def question_images(image: np.ndarray, mask: np.ndarray, region: str, max_side: int = 768) -> list[Image.Image]:
    full = Image.fromarray(image)
    box = region_box(mask, region)
    if box is None:
        full.thumbnail((max_side, max_side))
        return [full]
    marked = full.copy()
    lw = max(2, int(0.006 * max(full.size)))
    ImageDraw.Draw(marked).rectangle(box, outline=(255, 0, 0), width=lw)
    crop = full.crop(box)
    marked.thumbnail((max_side, max_side))
    crop.thumbnail((max_side // 2, max_side // 2))
    return [marked, crop]


class VLMScorer:
    """P(Yes) for a yes/no question, read from the first-token logits of a chat VLM (Qwen2.5-VL by default)."""

    def __init__(self, bundle):
        self.b = bundle
        tok = bundle.processor.tokenizer
        self.yes_ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", "yes", " Yes")})
        self.no_ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in ("No", "no", " No")})

    def p_yes(self, images: list[Image.Image], question: str) -> float:
        import torch

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question + " Answer with Yes or No only."})
        messages = [{"role": "user", "content": content}]
        proc, model = self.b.processor, self.b.model
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=images, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.b.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        ly = torch.logsumexp(logits[self.yes_ids], 0)
        ln = torch.logsumexp(logits[self.no_ids], 0)
        return float(torch.sigmoid(ly - ln).item())
