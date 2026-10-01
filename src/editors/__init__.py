"""Editors. Heavy imports (diffusers) happen only when a pipeline is actually loaded."""

from __future__ import annotations

from src.editors.base import Editor, ParamRange


def build_editor(name: str, cfg: dict, registry) -> Editor:
    if name == "ip2p":
        from src.editors.ip2p import InstructPix2PixEditor

        return InstructPix2PixEditor(cfg, registry)
    if name == "compositing":
        from src.editors.compositing import CompositingEditor

        return CompositingEditor(cfg, registry)
    raise ValueError(f"Unknown editor '{name}' (ip2p | compositing)")


__all__ = ["Editor", "ParamRange", "build_editor"]
