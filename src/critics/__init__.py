"""Critic registry: name -> class. Build the active set from config."""

from __future__ import annotations

from src.critics.base import Critic, CriticContext
from src.critics.follow import BackgroundSemanticCritic, OldBackgroundResidueCritic
from src.critics.gate import BackgroundChangedCritic, IdentityCritic, SubjectCountCritic
from src.critics.keep import AppearanceCritic, SilhouetteCritic, TextureCritic
from src.critics.vlm import VLM_CRITICS
from src.critics.world import ContactShadowCritic, HaloCritic, LightHarmonyCritic, SupportCritic

CRITICS: dict[str, type[Critic]] = {c.name: c for c in [
    SubjectCountCritic, IdentityCritic, BackgroundChangedCritic,
    SilhouetteCritic, AppearanceCritic, TextureCritic,
    BackgroundSemanticCritic, OldBackgroundResidueCritic,
    SupportCritic, ContactShadowCritic, LightHarmonyCritic, HaloCritic, *VLM_CRITICS,
]}


def build_critics(cfg: dict) -> list[Critic]:
    ccfg = cfg.get("critics", {})
    names = ccfg.get("enabled", list(CRITICS))
    unknown = [n for n in names if n not in CRITICS]
    if unknown:
        raise ValueError(f"Unknown critics {unknown}. Known: {sorted(CRITICS)}")
    return [CRITICS[n](**ccfg.get(n, {})) for n in names]


__all__ = ["CRITICS", "Critic", "CriticContext", "build_critics"]
