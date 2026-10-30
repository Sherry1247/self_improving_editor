"""RegionRepair: local, region-bound repair of an already edited image (innovation F).

The critics say WHERE a result is physically wrong (contact band C / boundary band B). Instead of
re-editing the whole image, inpaint only that region with the subject pixels frozen, then re-score.

Variants (so a gain cannot be blamed on "more compute" or "any extra inpainting"):
    A  none         the edited image as is
    B  contact      repair C only
    C  boundary     repair B only
    D  both         repair C + B
    E  misplaced    repair an equal-area region of the same shape placed away from the subject
(F full-image qwen re-edit and G best-of-N reseed come from the existing editor runs.)

Contact region shapes (``kind``):
    rect       legacy axis-aligned box under the feet (regions.partition)
    footprint  lowest part of the silhouette projected down column by column (follows the foot outline)
    cast       footprint sheared along a light direction (cast-shadow wedge)
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.geometry import dilate, erode, mask_box

PROMPT_CONTACT = ("a soft realistic contact shadow and ground contact under the {subject}, "
                  "consistent {bg} ground, natural lighting, photo")
PROMPT_BOUNDARY = "clean natural edge between the {subject} and the {bg} scene, no halo, consistent lighting, photo"
NEGATIVE = "halo, outline, cutout, floating, blurry, deformed, extra limbs, duplicate, text, watermark"


def footprint_region(subject: np.ndarray, foot_frac: float = 0.12, depth_frac: float = 0.10,
                     light_dx: float = 0.0) -> np.ndarray:
    """Shadow-shaped contact region: project the lowest ``foot_frac`` of the silhouette downward
    (``depth_frac`` * height), shear by ``light_dx`` px per px of depth, widen slightly, exclude subject."""
    h, w = subject.shape
    box = mask_box(subject)
    if box is None:
        return np.zeros((h, w), bool)
    _, y1, _, y2 = box
    sh = y2 - y1
    foot = subject.copy()
    foot[: int(y2 - foot_frac * sh)] = False
    depth = max(2, int(depth_frac * sh))
    out = np.zeros((h, w), np.uint8)
    for x in np.where(foot.any(axis=0))[0]:
        bottom = np.where(foot[:, x])[0].max()
        for d in range(depth + 1):
            xx, yy = int(round(x + light_dx * d)), bottom + d
            if 0 <= xx < w and 0 <= yy < h:
                out[yy, xx] = 1
    k = max(3, (int(0.02 * sh) // 2) * 2 + 1)
    out = cv2.dilate(out, np.ones((k, k), np.uint8)) > 0
    out |= dilate(foot, 2) & ~subject  # keep the immediate contact seam
    return out & ~subject


def contact_region(subject: np.ndarray, rect: np.ndarray, kind: str = "rect", light_dx: float = 0.0) -> np.ndarray:
    if kind == "rect":
        return rect & ~subject
    if kind == "footprint":
        return footprint_region(subject)
    if kind == "cast":
        return footprint_region(subject, depth_frac=0.18, light_dx=light_dx)
    raise ValueError(f"unknown contact kind {kind!r}")


def misplaced_region(target: np.ndarray, subject: np.ndarray, avoid: np.ndarray, seed: int = 0,
                     tries: int = 300) -> np.ndarray | None:
    """Translate ``target`` so it keeps its exact area/shape but lies away from the subject and ``avoid``.
    None if no placement exists (caller skips this control for the task)."""
    h, w = target.shape
    ys, xs = np.where(target)
    if len(ys) == 0:
        return None
    rng = np.random.default_rng(seed)
    forbid = dilate(subject | avoid, max(3, int(0.03 * min(h, w))))
    for _ in range(tries):
        dy = int(rng.integers(-ys.min(), h - ys.max()))
        dx = int(rng.integers(-xs.min(), w - xs.max()))
        ny, nx = ys + dy, xs + dx
        if forbid[ny, nx].any():
            continue
        out = np.zeros((h, w), bool)
        out[ny, nx] = True
        return out
    return None


def band_region(boundary_inner: np.ndarray, boundary_outer: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Boundary band B: outer half plus a thin inner rim; the subject core stays frozen."""
    rim = boundary_inner & ~erode(subject, 1)
    return (boundary_outer | rim) & ~erode(subject, 2)


class RegionRepairer:
    """Inpaint one region with SDXL, then keep every pixel outside the (feathered) region and the subject."""

    def __init__(self, cfg: dict, registry):
        self.cfg = cfg.get("region_repair", {})
        self.reg = registry

    def inpaint(self, work: np.ndarray, mask: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        import torch
        from PIL import Image

        h, w = work.shape[:2]
        with self.reg.use("sdxl_inpaint") as b:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            out = b.model(prompt=prompt, negative_prompt=NEGATIVE, image=Image.fromarray(work),
                          mask_image=Image.fromarray((mask * 255).astype(np.uint8)), height=h, width=w,
                          strength=float(self.cfg.get("strength", 0.6)),
                          guidance_scale=float(self.cfg.get("guidance_scale", 7.0)),
                          num_inference_steps=int(self.cfg.get("num_inference_steps", 30)), generator=gen).images[0]
        return np.asarray(out.convert("RGB"))

    def repair(self, image: np.ndarray, region: np.ndarray, subject: np.ndarray, prompt: str, seed: int = 0) -> np.ndarray:
        from src.data.images import match_size, resize_long_side

        if not region.any():
            return image.copy()
        work = resize_long_side(image, int(self.cfg.get("long_side", 1024)))
        h, w = work.shape[:2]
        rs = cv2.resize(dilate(region, 2).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out = match_size(self.inpaint(work, rs, prompt, seed), image.shape[:2])
        return blend_region(image, out, region, subject, float(self.cfg.get("feather_px", 2.0)))


def blend_region(image: np.ndarray, generated: np.ndarray, region: np.ndarray, subject: np.ndarray,
                 feather_px: float = 2.0) -> np.ndarray:
    """Only the feathered region may change; the subject is hard-frozen; elsewhere the input is bit-exact."""
    a = np.clip(cv2.GaussianBlur(dilate(region, 2).astype(np.float32), (0, 0), feather_px), 0, 1)
    a = a * (~subject) * dilate(region, 2 + int(3 * feather_px))
    res = a[..., None] * generated + (1 - a[..., None]) * image
    return res.astype(np.uint8)


def build_variants(image: np.ndarray, regions, subject_name: str, bg_name: str, repairer: RegionRepairer,
                   kind: str = "rect", light_dx: float = 0.0, seed: int = 0) -> dict[str, dict]:
    """{variant: {"image", "region_px"}} for A..E; ``regions`` belong to the edited image."""
    S = regions.subject
    C = contact_region(S, regions.contact, kind, light_dx)
    B = band_region(regions.boundary_inner, regions.boundary_outer, S)
    pc = PROMPT_CONTACT.format(subject=subject_name, bg=bg_name)
    pb = PROMPT_BOUNDARY.format(subject=subject_name, bg=bg_name)
    out = {"A_none": {"image": image.copy(), "region_px": 0},
           "B_contact": {"image": repairer.repair(image, C, S, pc, seed), "region_px": int(C.sum())},
           "C_boundary": {"image": repairer.repair(image, B, S, pb, seed), "region_px": int(B.sum())}}
    CB = C | B
    out["D_both"] = {"image": repairer.repair(image, CB, S, pc, seed), "region_px": int(CB.sum())}
    M = misplaced_region(C, S, B | C, seed)
    if M is not None:
        out["E_misplaced"] = {"image": repairer.repair(image, M, S, pc, seed), "region_px": int(M.sum())}
    return out
