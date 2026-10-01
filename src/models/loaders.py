"""One loader per model kind. Imports are lazy so tests never need the heavy libraries.

Loading failures RAISE. The legacy code silently fell back to mock detectors when the real
model failed (it always did: transformers 5.x renamed ``box_threshold`` / ``iou_predictions``),
which made every score meaningless. Mock perception now only exists in tests.
"""

from __future__ import annotations

from src.models.registry import ModelBundle


def load_grounding_dino(mcfg: dict, dtype, device: str) -> ModelBundle:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(mcfg["id"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mcfg["id"], dtype=dtype).eval()
    return ModelBundle("grounding_dino", mcfg["id"], model, proc)


def load_sam2(mcfg: dict, dtype, device: str) -> ModelBundle:
    from transformers import Sam2Model, Sam2Processor

    proc = Sam2Processor.from_pretrained(mcfg["id"])
    model = Sam2Model.from_pretrained(mcfg["id"], dtype=dtype).eval()
    return ModelBundle("sam2", mcfg["id"], model, proc)


def load_dinov2(mcfg: dict, dtype, device: str) -> ModelBundle:
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(mcfg["id"])
    model = AutoModel.from_pretrained(mcfg["id"], dtype=dtype).eval()
    return ModelBundle("dinov2", mcfg["id"], model, proc)


def load_siglip(mcfg: dict, dtype, device: str) -> ModelBundle:
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(mcfg["id"])
    model = AutoModel.from_pretrained(mcfg["id"], dtype=dtype).eval()
    return ModelBundle("siglip", mcfg["id"], model, proc)


def load_depth(mcfg: dict, dtype, device: str) -> ModelBundle:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(mcfg["id"])
    model = AutoModelForDepthEstimation.from_pretrained(mcfg["id"], dtype=dtype).eval()
    return ModelBundle("depth", mcfg["id"], model, proc)


def load_ip2p(mcfg: dict, dtype, device: str) -> ModelBundle:
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(mcfg["id"], torch_dtype=dtype, safety_checker=None,
                                                                  requires_safety_checker=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return ModelBundle("ip2p", mcfg["id"], pipe)


def load_sdxl_inpaint(mcfg: dict, dtype, device: str) -> ModelBundle:
    from diffusers import AutoPipelineForInpainting

    kwargs = {"torch_dtype": dtype}
    if str(dtype).endswith("float16"):
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForInpainting.from_pretrained(mcfg["id"], **kwargs)
    pipe.set_progress_bar_config(disable=True)
    offload = bool(mcfg.get("cpu_offload", False)) and device == "cuda"
    if offload:
        pipe.enable_model_cpu_offload()
    return ModelBundle("sdxl_inpaint", mcfg["id"], pipe, self_offloading=offload)


LOADERS = {
    "grounding_dino": load_grounding_dino,
    "sam2": load_sam2,
    "dinov2": load_dinov2,
    "siglip": load_siglip,
    "depth": load_depth,
    "ip2p": load_ip2p,
    "sdxl_inpaint": load_sdxl_inpaint,
}
