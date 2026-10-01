"""Load-once model registry with a per-model GPU memory policy.

Policies (``memory_policy`` globally, or ``models.<kind>.policy`` per model):

* ``keep``   — load straight onto the GPU and leave it there. No host-RAM copy. Default.
* ``swap``   — keep on CPU, move to GPU only inside ``registry.use(...)``. Saves VRAM but needs
               host RAM for every model (on Windows this also eats commit / page-file budget).
* ``unload`` — load on use, delete right after. Minimal memory, slowest.

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
        self.default_policy = cfg.get("memory_policy", "keep")
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

    def policy(self, kind: str) -> str:
        if self.device == "cpu":
            return "keep"
        return self.model_cfg(kind).get("policy", self.default_policy)

    def get(self, kind: str) -> ModelBundle:
        mcfg = self.model_cfg(kind)
        key = (kind, mcfg["id"], self.device)
        if key not in self._cache:
            dtype = resolve_dtype(mcfg.get("dtype", "float32"), self.device)
            direct = self.policy(kind) in ("keep", "unload")
            logger.info("Loading %s (%s, %s, policy=%s)", kind, mcfg["id"], dtype, self.policy(kind))
            bundle = self._loaders[kind](mcfg, dtype=dtype, device=self.device if direct else "cpu")
            bundle.dtype = dtype
            if direct or bundle.self_offloading:
                bundle.to(self.device)
            self._cache[key] = bundle
            self.log_memory(f"after loading {kind}")
        return self._cache[key]

    @contextmanager
    def use(self, kind: str) -> Iterator[ModelBundle]:
        policy = self.policy(kind)
        if policy == "swap":  # make room: push other swap-policy models back to CPU
            for (k, _, _), other in self._cache.items():
                if other.location != "cpu" and self.policy(k) == "swap" and k != kind:
                    other.to("cpu")
            self._free()
        bundle = self.get(kind)
        bundle.to(self.device)
        try:
            yield bundle
        finally:
            try:
                if policy == "swap":
                    bundle.to("cpu")
                elif policy == "unload":
                    self.unload(kind)
                self._free()
            except Exception as e:  # never mask the original error with a cleanup error
                logger.error("cleanup after %s failed: %s", kind, e)

    def unload(self, kind: str) -> None:
        for key in [k for k in self._cache if k[0] == kind]:
            del self._cache[key]
        self._free()

    def loaded(self) -> list[str]:
        return [k[0] for k in self._cache]

    def log_memory(self, where: str = "") -> None:
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.info("GPU memory %s: %.2f / %.2f GB used (torch allocated %.2f GB)", where,
                            (total - free) / 2**30, total / 2**30, torch.cuda.memory_allocated() / 2**30)
        except Exception:
            pass

    @staticmethod
    def _free() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
