"""Perception: one pass of all vision models over ONE image, producing a :class:`Perception`.

Models run one at a time through ``registry.use(...)`` so the whole stack fits in 8 GB.
Before-images are cached on disk (they never change across iterations / targets).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.models import ModelRegistry
from src.spec import BACKGROUNDS, EditSpec
from src.types import Detection, Perception
from src.utils.geometry import dilate, largest_component, nms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BG_KEYS = list(BACKGROUNDS)  # SigLIP always scores against every known background


class Perceiver:
    def __init__(self, registry: ModelRegistry, cfg: dict):
        self.reg = registry
        self.pcfg = cfg.get("perception", {})
        self.cache_dir = Path(cfg.get("paths", {}).get("cache", "data/cache")) / "perception"
        self._cfg_hash = hashlib.md5(json.dumps(
            {k: cfg["models"][k]["id"] for k in ("grounding_dino", "sam2", "dinov2", "siglip", "depth")}
            | {"p": self.pcfg}, sort_keys=True).encode()).hexdigest()[:8]

    # ------------------------------------------------------------------ public
    def perceive(self, image: np.ndarray, spec: EditSpec) -> Perception:
        h, w = image.shape[:2]
        p = self.pcfg
        dets = self.detect(image, spec.subject_query + " .", p.get("subject_box_threshold", 0.35),
                           p.get("subject_text_threshold", 0.25))
        min_area = p.get("subject_min_area_frac", 0.005) * h * w
        dets = [d for d in dets if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) >= min_area]
        keep = nms([d.box for d in dets], [d.score for d in dets], p.get("subject_nms_iou", 0.6))
        dets = sorted([dets[i] for i in keep], key=lambda d: -d.score)

        if dets:
            masks = self.segment(image, [d.box for d in dets])
            for d, m in zip(dets, masks):
                d.mask = m
            subject_mask = largest_component(dets[0].mask)
        else:
            subject_mask = np.zeros((h, w), bool)

        old_dets = self.detect(image, spec.old_bg_query, p.get("old_bg_box_threshold", 0.30), 0.25)
        old_mask = np.zeros((h, w), bool)
        if old_dets:
            for m in self.segment(image, [d.box for d in old_dets]):
                old_mask |= m
        old_mask &= ~dilate(subject_mask, 3)

        emb, patches = self.dino(image, subject_mask)
        depth = self.depth(image) if p.get("use_depth", True) else None
        bg_probs = self.background_probs(remove_subject(image, subject_mask))
        return Perception(image=image, subject_detections=dets, subject_mask=subject_mask, old_bg_mask=old_mask,
                          depth=depth, subject_embedding=emb, patch_features=patches, bg_probs=bg_probs)

    def perceive_cached(self, image: np.ndarray, spec: EditSpec, key: str) -> Perception:
        """Cache keyed by sample + resolution + model ids. Valid because old-bg concepts depend only on the source."""
        path = self.cache_dir / f"{key}_{image.shape[0]}x{image.shape[1]}_{self._cfg_hash}.npz"
        if path.exists():
            return load_perception(path, image)
        per = self.perceive(image, spec)
        save_perception(per, path)
        return per

    # ------------------------------------------------------------- components
    @torch.inference_mode()
    def detect(self, image: np.ndarray, query: str, box_thr: float, text_thr: float) -> list[Detection]:
        with self.reg.use("grounding_dino") as b:
            dev = next(b.model.parameters()).device
            size = self.pcfg.get("detector_size", {"shortest_edge": 512, "longest_edge": 800})
            inputs = b.processor(images=Image.fromarray(image), text=query.lower(), size=size,
                                 return_tensors="pt").to(dev)
            out = b.model(**inputs)
            res = b.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
                target_sizes=[image.shape[:2]])[0]
        labels = res.get("text_labels", res.get("labels"))
        return [Detection(label=str(l), box=tuple(float(v) for v in bx.tolist()), score=float(s))
                for bx, s, l in zip(res["boxes"], res["scores"], labels)]

    @torch.inference_mode()
    def segment(self, image: np.ndarray, boxes: list) -> list[np.ndarray]:
        if not boxes:
            return []
        masks = []
        with self.reg.use("sam2") as b:
            dev = next(b.model.parameters()).device
            pil = Image.fromarray(image)
            for box in boxes:
                inputs = b.processor(images=pil, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(dev)
                if b.dtype is not None and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(b.dtype)
                out = b.model(**inputs, multimask_output=True)
                post = b.processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
                cand = post[0]  # (num_masks, H, W)
                best = int(torch.argmax(out.iou_scores[0, 0]).item())
                masks.append(cand[best].numpy().astype(bool))
        return masks

    @torch.inference_mode()
    def dino(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """(CLS embedding of the masked subject crop, L2-normalised patch grid of the full image)."""
        with self.reg.use("dinov2") as b:
            dev = next(b.model.parameters()).device
            patches = _dino_forward(b.model, _dino_tensor(image, 518), dev, b.dtype)[1]
            emb = None
            crop = masked_crop(image, mask)
            if crop is not None:
                emb = _dino_forward(b.model, _dino_tensor(crop, 224, square=True), dev, b.dtype)[0]
        return emb, patches

    @torch.inference_mode()
    def depth(self, image: np.ndarray) -> np.ndarray:
        with self.reg.use("depth") as b:
            dev = next(b.model.parameters()).device
            inputs = b.processor(images=Image.fromarray(image), return_tensors="pt").to(dev)
            inputs["pixel_values"] = inputs["pixel_values"].to(b.dtype)
            pred = b.model(**inputs).predicted_depth.float()
            pred = torch.nn.functional.interpolate(pred[:, None], size=image.shape[:2], mode="bicubic",
                                                   align_corners=False)[0, 0].cpu().numpy()
        lo, hi = np.percentile(pred, 1), np.percentile(pred, 99)
        return np.clip((pred - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    @torch.inference_mode()
    def background_probs(self, bg_image: np.ndarray) -> dict[str, float]:
        texts = [f"a photo of a {BACKGROUNDS[k].name}" for k in BG_KEYS]
        with self.reg.use("siglip") as b:
            dev = next(b.model.parameters()).device
            inputs = b.processor(text=texts, images=Image.fromarray(bg_image), padding="max_length",
                                 return_tensors="pt").to(dev)
            inputs["pixel_values"] = inputs["pixel_values"].to(b.dtype)
            logits = b.model(**inputs).logits_per_image[0].float()
            probs = torch.softmax(logits, -1).cpu().numpy()
        return {k: float(p) for k, p in zip(BG_KEYS, probs)}


# ---------------------------------------------------------------------- helpers
def remove_subject(image: np.ndarray, mask: np.ndarray, work: int = 384) -> np.ndarray:
    """Cheap subject removal (OpenCV Telea inpaint at low res) so background classifiers see only the scene."""
    if not mask.any():
        return image
    h, w = image.shape[:2]
    s = work / max(h, w)
    small = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    m = cv2.resize(dilate(mask, max(2, int(0.02 * max(h, w)))).astype(np.uint8), small.shape[1::-1],
                   interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)


def masked_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.08) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    h, w = mask.shape
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    crop = image[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = (IMAGENET_MEAN * 255).astype(np.uint8)  # neutral fill = zero after normalisation
    return crop


def _dino_tensor(img: np.ndarray, size: int, square: bool = False) -> torch.Tensor:
    h, w = img.shape[:2]
    if square:
        nh = nw = size
    else:
        s = size / max(h, w)
        nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
    x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def _dino_forward(model, x: torch.Tensor, dev, dtype) -> tuple[np.ndarray, np.ndarray]:
    gh, gw = x.shape[2] // 14, x.shape[3] // 14
    hs = model(pixel_values=x.to(dev, dtype)).last_hidden_state.float()
    cls = torch.nn.functional.normalize(hs[0, 0], dim=-1).cpu().numpy()
    n_reg = hs.shape[1] - 1 - gh * gw  # handles register-token variants
    patches = hs[0, 1 + n_reg:].reshape(gh, gw, -1)
    patches = torch.nn.functional.normalize(patches, dim=-1).cpu().numpy()
    return cls, patches


# ---------------------------------------------------------------- persistence
def save_perception(p: Perception, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dets = [{"label": d.label, "box": list(d.box), "score": d.score} for d in p.subject_detections]
    det_masks = np.stack([d.mask for d in p.subject_detections]) if p.subject_detections else np.zeros((0, 1, 1), bool)
    np.savez_compressed(
        path, subject_mask=p.subject_mask, old_bg_mask=p.old_bg_mask, det_masks=det_masks,
        depth=p.depth if p.depth is not None else np.zeros(0),
        subject_embedding=p.subject_embedding if p.subject_embedding is not None else np.zeros(0),
        patch_features=p.patch_features.astype(np.float16) if p.patch_features is not None else np.zeros(0),
        meta=json.dumps({"dets": dets, "bg_probs": p.bg_probs}),
    )


def load_perception(path: Path, image: np.ndarray) -> Perception:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    dets = [Detection(d["label"], tuple(d["box"]), d["score"], z["det_masks"][i] if len(z["det_masks"]) else None)
            for i, d in enumerate(meta["dets"])]
    opt = lambda k: z[k] if z[k].size else None  # noqa: E731
    pf = opt("patch_features")
    return Perception(image=image, subject_detections=dets, subject_mask=z["subject_mask"], old_bg_mask=z["old_bg_mask"],
                      depth=opt("depth"), subject_embedding=opt("subject_embedding"),
                      patch_features=pf.astype(np.float32) if pf is not None else None, bg_probs=meta["bg_probs"])
