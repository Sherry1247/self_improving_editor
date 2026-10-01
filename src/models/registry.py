"""Load-once model registry with an 8 GB-friendly GPU memory policy.

``memory_policy: swap`` keeps every model on the CPU and moves only the one in use
onto the GPU (``with registry.use("sam2") as m: ...``). ``keep`` leaves everything on the GPU.
Cache key is (kind, model_id, device) — fixes the legacy CLIP cache that ignored model_id.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(name: str, device: str):
    import torch

    if device == "cpu":
        return torch.float32  # half precision on CPU is slow / unsupported for many ops
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


@dataclass
class ModelBundle:
    kind: str
    model_id: str
    model: Any  # nn.Module or diffusers pipeline
    processor: Any = None
    dtype: Any = None
    self_offloading: bool = False  # diffusers enable_model_cpu_offload manages its own placement
    location: str = "cpu"

    def to(self, device: str) -> None:
        if self.self_offloading or self.location == device:
            return
        self.model.to(device)
        self.location = device


class ModelRegistry:
    def __init__(self, cfg: dict, loaders: dict[str, Callable[..., ModelBundle]] | None = None):
        from src.models import loaders as default_loaders

        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        self.policy = cfg.get("memory_policy", "swap")
        self._loaders = loaders or default_loaders.LOADERS
        self._cache: dict[tuple[str, str, str], ModelBundle] = {}
        if cfg.get("paths", {}).get("hf_home"):
            import os

            os.environ["HF_HOME"] = cfg["paths"]["hf_home"]

    def model_cfg(self, kind: str) -> dict:
        try:
            return self.cfg["models"][kind]
        except KeyError as e:
            raise KeyError(f"No config for model '{kind}' under models:") from e

    def get(self, kind: str) -> ModelBundle:
        mcfg = self.model_cfg(kind)
        key = (kind, mcfg["id"], self.device)
        if key not in self._cache:
            dtype = resolve_dtype(mcfg.get("dtype", "float32"), self.device)
            logger.info("Loading %s (%s, %s)", kind, mcfg["id"], dtype)
            bundle = self._loaders[kind](mcfg, dtype=dtype, device=self.device)
            bundle.dtype = dtype
            if self.policy == "keep" or bundle.self_offloading:
                bundle.to(self.device)
            self._cache[key] = bundle
        return self._cache[key]

    @contextmanager
    def use(self, kind: str) -> Iterator[ModelBundle]:
        bundle = self.get(kind)
        if self.policy == "swap":
            for other in self._cache.values():
                if other is not bundle and other.location != "cpu":
                    other.to("cpu")
            self._free()
        bundle.to(self.device)
        try:
            yield bundle
        finally:
            if self.policy == "swap":
                bundle.to("cpu")
                self._free()

    def unload(self, kind: str) -> None:
        for key in [k for k in self._cache if k[0] == kind]:
            del self._cache[key]
        self._free()

    def loaded(self) -> list[str]:
        return [k[0] for k in self._cache]

    @staticmethod
    def _free() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
