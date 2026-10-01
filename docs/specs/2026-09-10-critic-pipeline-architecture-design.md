# Critic Pipeline Architecture — Design Spec

*Date: 2026-09-10. Status: proposed, pending user review. Scope: Phase 2 (architecture) + enough of Phase 3 (migration categorization) to hand off to an implementation plan. Does NOT decide aggregation math, object-matching algorithm details, or instruction-conditioning logic — those are a follow-up spec (see "Explicitly deferred" below).*

## 1. Problem this solves

Full findings are in [`PHASE1_AUDIT.md`](../../../PHASE1_AUDIT.md) (verified accurate against the repo as of this date — no commits since it was written). Summary: the repo contains two disconnected systems —

- **System A** (`main.py` / `ClosedLoopPipeline`): the only system that actually runs end-to-end. Single-object YOLO detection, Pix2Pix editing, 3 shallow critics, flat weighted sum.
- **System B** (`experiments/run_evaluation.py`): a richer, structurally-correct-for-the-research-brief critic stack (Grounding DINO, SAM2, scene graph, 7 critics, reward aggregator, prompt refiner) that is never wired to an editor and defaults to `use_mock: true`, making its scores currently meaningless.

Beyond the disconnect, the audit found: `bbox_iou` implemented 3×, weighted-sum aggregation implemented 3×, two incompatible prompt-refinement modules, redundant critic re-scoring that doubles compute per iteration, and a CLIP-model-cache bug that ignores `model_id`.

**Decision (confirmed with user):** integrate into one pipeline rather than keep the systems separate. Real Grounding DINO/SAM2/CLIP weights are available; no VLM API is wired yet, so the semantic critic must work meaningfully on CLIP alone for now and treat VLM as an optional, clearly-labeled add-on. The editor becomes an abstracted interface (Pix2Pix stays the only implementation for now).

## 2. Goals / non-goals

**Goals:**
- One code path serves both "evaluate a single before/after pair" (System B's use case) and "iteratively edit and refine" (System A's use case), sharing all detection/scoring logic.
- Every model (Grounding DINO, SAM2, CLIP, VLM) loads once per process via a registry, keyed correctly (fixes the `model_id` cache bug).
- Every geometry/IO/aggregation helper exists in exactly one place.
- Mock mode, if used, is impossible to miss in output artifacts.
- The design supports (without yet implementing) instruction-conditioned evaluation, object-role tagging, and pluggable aggregation — because Phase 5 needs to build those on top without another restructure.

**Non-goals for this spec:**
- Deciding the aggregation formula (weighted sum vs hierarchical gating vs Pareto vs learned) — Section 6 leaves an `Aggregator` seam but does not fill it in.
- Deciding the object-matching algorithm's exact scoring function (Hungarian cost matrix weights) — the seam (`match_objects`) is defined, the algorithm is not.
- Adding a real VLM integration or a new editor implementation.
- Experiment/ablation harness design (Phase 6).

## 3. Chosen approach

Three integration approaches were considered:

| | A. Big-bang replacement | B. Config-driven registry, single loop | C. Evaluator + Loop split (chosen) |
|---|---|---|---|
| Ablation support (Experiment 3: which critic matters) | Poor — old critics deleted, nothing to compare against | Good | Good |
| Standalone scoring (System B's current use case) | N/A, folded into loop | Requires spinning up loop machinery | Native — `Evaluator` alone |
| Migration risk | High, all-or-nothing | Medium | Medium, but splits into independently-landable pieces |

**Chosen: C.** Split into a standalone `Evaluator` (image-before + image-after + instruction → `EvaluationResult`) and a thin `RefinementLoop` orchestrator that calls `Editor` + `Evaluator` + `PromptRefiner` across iterations. Critics are config-driven (a registry keyed by name), so both direction-1-style single-shot evaluation and direction-2-style iterative refinement are the same underlying code, run differently.

## 4. Directory layout

```
src/
  models/                 # shared model registry — load-once wrappers
    registry.py           #   ModelRegistry: get("grounding_dino"|"sam2"|"clip"|"vlm")
    grounding_dino.py      #   moved from detectors/, logic unchanged
    sam2.py                 #   moved from segmentation/, logic unchanged
    clip_backbone.py
    vlm.py                  #   optional; absent/mocked backend is fine, but tagged in output
  editors/
    base.py                 # Editor interface (kept as-is)
    pix2pix_editor.py
  data/
    types.py                # Detection, ObjectCorrespondence, SceneGraph, CriticResult, EvaluationResult
  scene/
    scene_graph.py           # SceneGraphBuilder, moved from scene_graph/
    object_matching.py        # NEW — Hungarian correspondence (algorithm: Phase 5)
  critics/
    base.py                  # Critic interface
    detection.py               # merges detection.py + count.py
    segmentation.py
    spatial.py
    semantic.py                 # merges clip*.py + vlm.py; reports subject-preservation AND
                                 #   instruction-adherence as separate sub-scores, not one blended number
    physics.py
    registry.py                  # name -> Critic class, instantiated from config
  scoring/
    aggregation.py             # NEW home for the aggregation strategy (Phase 5 fills this in)
  refinement/
    feedback_summarizer.py      # NEW — collapses/prioritizes issues across iterations
    prompt_refiner.py            # unifies prompts/refinement.py + prompt_refinement/refiner.py
  pipelines/
    evaluator.py                 # NEW — Evaluator
    refinement_loop.py            # NEW — RefinementLoop
  utils/
    geometry.py                   # NEW — single bbox_iou/mask_iou implementation
    image.py
    visualization.py
  config/
    models.yaml                   # device + model_id + checkpoint paths, one place
    critics.yaml                  # which critics are active + their weights/thresholds
experiments/
  run_evaluation.py               # thin CLI wrapper around pipelines.evaluator.Evaluator
  run_closed_loop.py               # thin CLI wrapper around pipelines.refinement_loop.RefinementLoop
```

`main.py`, `src/closed_loop_editor.py`, and `experiments/run_single.py` collapse into `experiments/run_closed_loop.py` — the ~90%-duplicated boilerplate (load config → build pipeline → load image → build prompt → run → write output) disappears because there's only one real implementation left for them to each wrap.

## 5. Shared model registry

Fixes the `model_id`-blind CLIP cache and the scattered `"cuda"`/`"cpu"`/hardcoded-path problem:

```python
@dataclass(frozen=True)
class ModelKey:
    kind: str        # "grounding_dino" | "sam2" | "clip" | "vlm"
    model_id: str    # part of the cache key — this is what today's clip_utils bug omits
    device: str

class ModelRegistry:
    def __init__(self, config: ModelConfig):
        self._cache: dict[ModelKey, Any] = {}
        self._config = config    # single source of device/model_id/checkpoint-path truth

    def get(self, kind: str) -> Any:
        key = ModelKey(kind, self._config.model_id(kind), self._config.device(kind))
        if key not in self._cache:
            self._cache[key] = _load(key, self._config)
        return self._cache[key]
```

One instance is constructed at pipeline startup and threaded down to detectors/critics. Nothing below `pipelines/` calls `torch.device(...)` or hardcodes a checkpoint path.

## 6. Core data structures (`src/data/types.py`)

```python
@dataclass
class Detection:
    label: str
    box: BBox                      # explicit xyxy convention, enforced at construction
    confidence: float
    mask: np.ndarray | None         # set once SAM2 runs; carried on the object, not re-fetched by index
    embedding: np.ndarray | None     # CLIP region embedding, computed once, reused by matching + semantic critic
    role: ObjectRole                  # PRIMARY_SUBJECT | INSTRUCTION_TARGET | SECONDARY | BACKGROUND_CONTEXT
                                       # (tagging logic is Phase 5; the field exists now so nothing downstream
                                       #  needs to change shape when instruction-conditioning lands)

@dataclass
class ObjectCorrespondence:
    before: Detection | None
    after: Detection | None
    match_score: float
    status: Literal["matched", "removed", "added"]

@dataclass
class SceneGraph:
    objects: list[Detection]
    relations: list[Relation]        # Relation(subject_idx, predicate, object_idx, confidence)

@dataclass
class CriticResult:
    score: float                      # [0, 1]
    issues: list[str]                  # human-readable, structured enough to feed PromptRefiner directly
    evidence: dict                      # raw numbers (IoU values, matched pairs, ...) for debugging/ablation logs
    is_catastrophic: bool = False        # this critic's own opinion on whether it should gate (Phase 5 consumes this)

@dataclass
class EvaluationResult:
    overall_score: float | None           # None until an Aggregator computes it — not baked into this schema
    critics: dict[str, CriticResult]        # keyed "detection" | "segmentation" | "spatial" | "semantic" | "physics"
    correspondences: list[ObjectCorrespondence]
    scene_graph_before: SceneGraph
    scene_graph_after: SceneGraph
    mock: bool                                # explicit — fixes the silent-mock problem (audit CRITICAL-1)
```

`Detection` carries its own mask and embedding rather than being recovered by string-parsing a scene-graph object id into a list index (the audit's MINOR-3). This is also what makes Hungarian matching possible without re-deriving geometry from scratch.

## 7. Critic interface

```python
class Critic(ABC):
    name: str

    @abstractmethod
    def evaluate(
        self,
        scene_before: SceneGraph,
        scene_after: SceneGraph,
        correspondences: list[ObjectCorrespondence],
        instruction: str,
    ) -> CriticResult: ...
```

All critics take the same pre-computed scene graphs and correspondences — detection, segmentation, and matching happen exactly once, upstream, inside `Evaluator`. No critic re-runs a detector or re-matches objects itself. This is the direct fix for the audit's CRITICAL-2/MAJOR-1 (redundant model calls).

## 8. Editor interface

```python
class Editor(ABC):
    @abstractmethod
    def edit(self, image: Image, instruction: str) -> Image: ...
```

`Pix2PixEditor` is the sole implementation. No behavior change; this just formalizes what's already an implicit contract so a future editor (e.g. InstructPix2Pix) is a new class, not a rewrite of `RefinementLoop`.

## 9. Pipeline orchestration

```python
class Evaluator:
    """image_before + image_after + instruction -> EvaluationResult. No editing, no iteration."""
    def __init__(self, registry: ModelRegistry, critics: list[Critic]): ...

    def evaluate(self, image_before, image_after, instruction) -> EvaluationResult:
        det_before = self._detect(image_before, instruction)     # detector + segmenter run ONCE per image
        det_after  = self._detect(image_after, instruction)
        correspondences = match_objects(det_before, det_after)     # Hungarian; algorithm is Phase 5
        scene_before, scene_after = build_scene_graph(det_before), build_scene_graph(det_after)
        results = {c.name: c.evaluate(scene_before, scene_after, correspondences, instruction)
                   for c in self._critics}
        return EvaluationResult(overall_score=None, critics=results,
                                 correspondences=correspondences,
                                 scene_graph_before=scene_before, scene_graph_after=scene_after,
                                 mock=self._registry.is_mock())

class RefinementLoop:
    """Drives Editor + Evaluator + PromptRefiner across iterations. Owns iteration/convergence policy only."""
    def __init__(self, editor: Editor, evaluator: Evaluator, aggregator: Aggregator,
                 refiner: PromptRefiner, max_iterations: int, score_threshold: float,
                 plateau_patience: int): ...

    def run(self, image, instruction) -> LoopResult:
        best = None
        history = []
        for i in range(self.max_iterations):
            edited = self.editor.edit(image, instruction)
            result = self.evaluator.evaluate(image, edited, instruction)
            result.overall_score = self.aggregator.aggregate(result.critics)
            history.append(result.overall_score)
            if best is None or result.overall_score > best.result.overall_score:
                best = Candidate(edited, result, i)
            if result.overall_score >= self.score_threshold:
                return LoopResult(best=best, all_candidates=history, stopped_reason="threshold_met")
            if _plateaued(history, self.plateau_patience):
                return LoopResult(best=best, all_candidates=history, stopped_reason="plateau")
            instruction = self.refiner.refine(instruction, result)   # summarized feedback, not appended forever
        return LoopResult(best=best, all_candidates=history, stopped_reason="max_iterations")
```

`run_evaluation.py`'s standalone use case and System A's closed loop become the same code: standalone evaluation is `Evaluator(...).evaluate(...)` called once; closed-loop editing is `RefinementLoop(...).run(...)`. Both share one detector pass, one critic registry, one aggregation strategy. `RefinementLoop` never sees critic internals or the aggregation formula — it only calls `self.aggregator.aggregate(...)`, which is the seam Phase 5 fills in. `LoopResult` always carries `best`, tracked independently of which iteration ran last (fixes the audit's observation that the final iteration is not guaranteed to be the best one).

## 10. Migration categorization (feeds the Phase 3/4 implementation plan)

| File | Action | Notes |
|---|---|---|
| `src/detectors/yolo_detector.py` | DELETE | Single-object-only; superseded by Grounding DINO once merged. Keep in git history, not in the new tree. |
| `src/detectors/grounding_dino.py` | REFACTOR → `src/models/grounding_dino.py` | Logic kept; loading goes through `ModelRegistry`. |
| `src/segmentation/sam2_segmentor.py` | REFACTOR → `src/models/sam2.py` | Same. |
| `src/critics/clip_utils.py` | MERGE into `src/models/clip_backbone.py` | Fold into registry; fixes the `model_id` cache bug as part of the move. |
| `src/critics/clip.py`, `clip_similarity.py`, `instruction_alignment.py`, `vlm.py` | MERGE → `src/critics/semantic.py` | One critic, two named sub-scores (subject preservation, instruction adherence) instead of 4 files each doing a slice of "CLIP similarity." |
| `src/critics/detection.py`, `count.py` | MERGE → `src/critics/detection.py` | Same input (correspondences), closely related questions (recall/precision/count-delta). |
| `src/critics/object_consistency.py` | DELETE | System-A-only top-1-IoU logic; superseded by the merged `detection.py` + `object_matching.py`. |
| `src/critics/segmentation.py`, `spatial.py`, `physics.py` | REFACTOR | Adopt the new `Critic` interface signature; internal logic mostly kept, revisited case-by-case during implementation (e.g. MINOR-1's hardcoded 384×384 fallback gets fixed here). |
| `src/critics/utils.py` (`_bbox_iou`, greedy matcher) | MERGE → `src/utils/geometry.py` (iou) + `src/scene/object_matching.py` (matcher, upgraded to Hungarian in Phase 5) | |
| `src/critics/composite.py`, `src/reward_aggregation/aggregator.py`, `src/utils/io_utils.py::combine_metrics` | DELETE, REWRITE → `src/scoring/aggregation.py` | Three implementations become one; formula itself is Phase 5. |
| `src/prompts/refinement.py`, `src/prompt_refinement/refiner.py` | MERGE → `src/refinement/prompt_refiner.py` | Must agree on one score-name vocabulary (today: `object_consistency/clip_similarity/instruction_alignment` vs `detection/count/segmentation/spatial/clip/vlm/physics` — the merged version uses the `EvaluationResult.critics` keys). |
| `src/scene_graph/builder.py` | REFACTOR → `src/scene/scene_graph.py` | Kept; relations gain depth-awareness caveats documented, not fixed, in this pass. |
| `src/pipelines/closed_loop_pipeline.py` | REWRITE → `src/pipelines/refinement_loop.py` | New shape per Section 9. |
| `experiments/run_evaluation.py` | REWRITE → thin CLI over `Evaluator` | |
| `main.py`, `src/closed_loop_editor.py`, `experiments/run_single.py` | DELETE, REWRITE → `experiments/run_closed_loop.py` | Collapses the 90%-duplicated boilerplate. |
| `src/editors/pix2pix_editor.py`, `src/editors/base.py` | KEEP | Already matches the target `Editor` interface. |
| `src/utils/device_utils.py`, `logging_config.py`, `visualization.py` | KEEP / minor REFACTOR | Device selection folds into `ModelConfig`; the rest is fine as-is. |
| `src/configs/evaluation.yaml` | REWRITE → `config/models.yaml` + `config/critics.yaml` | Split model config from critic-selection config; `use_mock` becomes a per-model override, not a single global flag silently defaulting true. |

**NEW files with no existing counterpart** (created fresh, not migrated from anything): `src/models/registry.py`, `src/data/types.py`, `src/scene/object_matching.py`, `src/critics/registry.py`, `src/scoring/aggregation.py` (module shell only — formula deferred), `src/refinement/feedback_summarizer.py`, `src/pipelines/evaluator.py`.

## 11. Explicitly deferred (follow-up specs)

- **Aggregation strategy** (weighted sum vs hierarchical gating vs Pareto vs learned) — `Aggregator` interface exists (Section 9), formula does not. This is Experiment 4 in the research brief and deserves its own design pass once the skeleton lands.
- **Object-matching algorithm** — `match_objects()` seam exists; the Hungarian cost function (label + IoU + centroid + CLIP region similarity + confidence weighting) is not specified here.
- **Instruction-conditioned evaluation / `ObjectRole` tagging logic** — the field exists on `Detection`; how it gets populated (primary subject vs background context, conditioned on the instruction) is not designed here.
- **Counterfactual / edit-locality critic, catastrophic-object-graph** — new critics under the existing `Critic` interface; not designed here, but the interface doesn't need to change to add them.
- **Real VLM wiring** — `models/vlm.py` is a slot; no provider is chosen here per your answer that no VLM API is available yet.
- **Experiments/ablation harness** (Phase 6).

## 12. Risks / open questions

- Merging `detection.py` + `count.py` and merging 4 CLIP/VLM files into `semantic.py` both reduce file count but increase per-file responsibility — if either grows past ~150-200 lines during implementation, re-split before it becomes another "does too much" file.
- The `Evaluator`/`RefinementLoop` split assumes `Editor.edit()` is stateless per call (no cross-iteration state needed by the editor itself) — true for Pix2Pix, worth re-checking if a future editor needs conversation-style state.
- Config split (`models.yaml` vs `critics.yaml`) needs a decision on where per-critic *thresholds used for gating* live once Phase 5 exists — likely `critics.yaml`, but that file doesn't have a settled schema yet since aggregation is deferred.
