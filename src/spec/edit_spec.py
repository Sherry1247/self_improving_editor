"""EditSpec: turns a dataset row + target background into a checkable contract.

The spec says what must stay the same (invariants), what must change (target),
and what must change *together with* the background (covariants). Every critic,
prompt and repair action is instantiated from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Background:
    key: str
    name: str  # short noun phrase used in prompts / VLM questions
    scene: str  # richer description for generative prompts
    support: str  # what the subject should stand / sit on
    concepts: tuple[str, ...]  # detector phrases that indicate this background
    is_water: bool = False
    may_contain: tuple[str, ...] = ()  # other concepts that can legitimately appear in this scene


BACKGROUNDS: dict[str, Background] = {
    "snow": Background("snow", "snowy field", "a snowy field in winter with snow-covered trees",
                       "snow-covered ground", ("snow",), may_contain=("mountain",)),
    "beach": Background("beach", "sandy beach", "a sunny sandy beach with the ocean in the distance",
                        "sand", ("sand", "beach"), may_contain=("water",)),
    "city": Background("city", "city street", "a city street with buildings and a sidewalk",
                       "paved sidewalk", ("building", "street")),
    "forest": Background("forest", "forest", "a green forest with tall trees",
                         "forest floor", ("tree", "forest"), may_contain=("mountain", "river", "water")),
    "indoor": Background("indoor", "indoor living room", "a cozy indoor living room with furniture",
                         "wooden floor", ("sofa", "furniture")),
    "mountain": Background("mountain", "mountain landscape", "a mountain landscape with rocky peaks",
                           "rocky ground", ("mountain",), may_contain=("snow", "tree")),
    "river": Background("river", "riverside", "a riverside with flowing water and rocks",
                        "riverbank rocks", ("river", "water"), is_water=True, may_contain=("tree", "mountain")),
}

# The five main targets + the swap (river <-> mountain) used for every image.
DEFAULT_TARGETS: tuple[str, ...] = ("snow", "beach", "city", "forest", "indoor")

SUBJECTS: dict[str, dict[str, str]] = {
    "adult_person": {"noun": "person", "query": "person"},
    "person": {"noun": "person", "query": "person"},
    "dog": {"noun": "dog", "query": "dog"},
}

POSES = {"sit": "sitting", "stand": "standing"}


@dataclass(frozen=True)
class EditSpec:
    sample_id: str
    subject: str  # noun, e.g. "dog"
    subject_query: str  # detector phrase
    pose: str  # "sitting" | "standing"
    source_bg: Background
    target_bg: Background
    submerged: bool = False  # subject is partly in water in the source image
    invariants: tuple[str, ...] = ("identity", "silhouette", "appearance")
    covariants: tuple[str, ...] = field(default_factory=tuple)

    @property
    def task_id(self) -> str:
        return f"{self.sample_id}__to_{self.target_bg.key}"

    @property
    def old_bg_may_reappear(self) -> bool:
        """True when the target scene can legitimately contain the source scene's concepts (e.g. river -> beach)."""
        allowed = set(self.target_bg.concepts) | set(self.target_bg.may_contain)
        return bool(set(self.source_bg.concepts) & allowed)

    @property
    def old_bg_query(self) -> str:
        return " . ".join(self.source_bg.concepts) + " ."

    def candidate_backgrounds(self) -> list[Background]:
        """Target first, then source, then the other default targets (for zero-shot classification)."""
        keys = [self.target_bg.key, self.source_bg.key] + list(DEFAULT_TARGETS) + ["mountain", "river"]
        seen: list[str] = []
        for k in keys:
            if k not in seen:
                seen.append(k)
        return [BACKGROUNDS[k] for k in seen]

    def instruction(self) -> str:
        """Instruction-style prompt (InstructPix2Pix / Kontext)."""
        return (f"replace the background with {self.target_bg.scene}, keep the {self.subject} "
                f"{self.pose} exactly the same")

    def scene_prompt(self) -> str:
        """Descriptive prompt (inpainting / text-to-image style)."""
        return (f"a photo of a {self.subject} {self.pose} on the {self.target_bg.support}, "
                f"{self.target_bg.scene}, natural lighting, realistic photo")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_id"] = self.task_id
        return d


def build_spec(sample_id: str, obj: str, action: str, background: str, target: str,
               submerged: bool = False) -> EditSpec:
    if obj not in SUBJECTS:
        raise ValueError(f"Unknown subject '{obj}'. Known: {sorted(SUBJECTS)}")
    if background not in BACKGROUNDS or target not in BACKGROUNDS:
        raise ValueError(f"Unknown background '{background}' or target '{target}'")
    if background == target:
        raise ValueError(f"{sample_id}: target background equals source background ({target})")
    src, tgt = BACKGROUNDS[background], BACKGROUNDS[target]
    covariants = ["support_contact", "contact_shadow", "light_harmony", "edge_clean"]
    if src.is_water and not tgt.is_water:
        covariants.append("reflection_removed")
    return EditSpec(
        sample_id=sample_id,
        subject=SUBJECTS[obj]["noun"],
        subject_query=SUBJECTS[obj]["query"],
        pose=POSES.get(action, action),
        source_bg=src,
        target_bg=tgt,
        submerged=submerged,
        covariants=tuple(covariants),
    )


def targets_for(source_bg: str, targets: tuple[str, ...] = DEFAULT_TARGETS, include_swap: bool = True) -> list[str]:
    out = [t for t in targets if t != source_bg]
    if include_swap:
        swap = {"river": "mountain", "mountain": "river"}.get(source_bg)
        if swap and swap not in out:
            out.append(swap)
    return out
