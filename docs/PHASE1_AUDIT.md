# Phase 1 Audit — `self_improving_editor`

*Audited: 2026-09-04. Scope: full repo (`src/`, `experiments/`, `chtc/`, `data/`, configs). No code changed in this pass.*

## 0. The single most important finding

**This repo contains two almost entirely separate systems that share a name but never talk to each other.**

**System A — "Legacy simple loop."** `main.py` / `src/closed_loop_editor.py` / `experiments/run_single.py` → `src/configs.build_pipeline()` → `ClosedLoopPipeline` (`src/pipelines/closed_loop_pipeline.py`) wired with `YOLODetector` + `Pix2PixEditor` + 3 critics (`ObjectConsistencyCritic`, `CLIPSimilarityCritic`, `InstructionAlignmentCritic`) combined by `CompositeCritic`. This is what the README documents, what `chtc/run.sh` submits to HTCondor, and what actually produces `data/experiments/*` and `data/metrics.csv`. **This is "production."**

**System B — "Direction-1 critic stack."** `experiments/run_evaluation.py` wires `GroundingDinoDetector`, `SAM2Segmentor`, `SceneGraphBuilder`, and 7 new critics (`DetectionCritic`, `CountCritic`, `SegmentationCritic`, `SpatialCritic`, `CLIPCritic`, `VLMCritic`, `PhysicsCritic`) through `RewardAggregator` and `PromptRefiner`. This is structurally very close to the "Direction 1" design in your brief (scene graph, structured feedback, per-critic issues). **It is not called from `main.py`, not called from `ClosedLoopPipeline`, not submitted by CHTC, and not mentioned in the README.** It is a standalone `argparse` script that scores one before/after image pair and writes JSON to `results/`. It never edits an image and never closes the loop — despite computing a `refined_instruction`, nothing feeds that back into an editor.

Every module in `src/critics/`, `src/detectors/`, `src/segmentation/`, `src/scene_graph/`, `src/reward_aggregation/`, `src/prompt_refinement/` that looks like it implements your research brief (Grounding DINO, SAM2, scene graph, physics critic, VLM critic, reward aggregation) belongs to System B, and System B is disconnected scaffolding, not a running pipeline. Before any refactor, you need to decide: **is System B the direction you're actually pursuing, or is it exploratory code that should be merged into a single pipeline?** Everything below assumes the answer is "yes, merge it" — that's Phase 2/3's job — but Phase 1 just reports what's there.

There's a second, even sharper problem inside System B: **its default config runs entirely on mock models.**

```yaml
# src/configs/evaluation.yaml
use_mock: true
```

`GroundingDinoDetector` in mock mode returns a bounding box that is a **fixed fraction of image width/height**, keyed only by the class name in the query string — it never looks at pixels. `SAM2Segmentor` in mock mode draws an ellipse inside that same box. `CLIPCritic` in mock mode returns the constant `0.78`. `VLMCritic` in mock mode returns the constant `0.82` with a templated reasoning string. Since `image_before` and `image_after` are (almost always) the same resolution, **detection, segmentation, count, and spatial critics will produce near-identical scene graphs for *any* before/after pair**, mock or real edit, good or catastrophic. Run today with defaults, `run_evaluation.py` cannot distinguish a perfect edit from a destroyed one — every score is a deterministic function of image size and the fixed class-name lookup table, not of what the editor actually did. This is flagged in detail in §3 (CRITICAL-1).

---

## 1. Current Pipeline — what actually runs (System A / production)

```
labels.csv (filename, object, action, background)
        │
        ▼
build_prompt()  →  "replace the current {bg} background with a different
                     scene, while keeping the {obj} and its {action} pose
                     unchanged, natural lighting, realistic photo"
        │
        ▼
┌─────────────────────────── ClosedLoopPipeline.run() ───────────────────────────┐
│  loop (max_iterations, default 3):                                             │
│                                                                                  │
│   original_image ──► Pix2PixEditor.edit(image, prompt) ──► edited_image        │
│                                                                                  │
│   CompositeCritic.score(original, edited, prompt)                              │
│     ├─ ObjectConsistencyCritic  : YOLOv8 top-1 box IoU (orig vs edited)        │
│     ├─ CLIPSimilarityCritic     : CLIP image↔image cosine similarity          │
│     └─ InstructionAlignmentCritic: CLIP image↔text cosine similarity          │
│     → composite = 0.4·obj_consistency + 0.3·clip_sim + 0.3·instr_align         │
│                                                                                  │
│   [duplicate work: get_individual_scores() re-runs all 3 critics again         │
│    just to log them — see MAJOR-1]                                             │
│   [duplicate work: detector.detect() called a 3rd time for visualization       │
│    — see MAJOR-1]                                                              │
│                                                                                  │
│   track best_image / best_score across iterations                              │
│   if composite ≥ score_threshold (0.7): stop early                             │
│   else: refine_prompt() appends a fixed clause per weak critic, loop again     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
data/experiments/<exp_id>/{metadata.json, iterations/NNN.json, images/*.jpg}
data/metrics.csv  (if --export-csv)
```

Object detection here is **single-object, top-1-confidence-only** (`YOLODetector.detect` returns one `(class_id, box)`, not a list) — it was built for "one person or one dog per image," which matches your 15-image, single-subject dataset, but has no path to multi-object scenes.

## 2. Current Pipeline — what's implemented but not wired up (System B)

```
run_evaluation.py --image-before --image-after --instruction --job-id
        │
        ▼
GroundingDinoDetector.detect(img, "person . dog . river . mountain . boat . fish . car .")
   (before AND after, independently)
        │
        ▼
SAM2Segmentor.segment(img, boxes)      (before AND after)
        │
        ▼
SceneGraphBuilder.build(detections, masks)  → {objects[], relationships[]}
   relationships inferred from box/mask geometry: inside, standing_on / floating_on, near
        │
        ▼ (before-graph, after-graph)
┌── critics (7, all independent, all read from the two scene graphs) ──┐
│ DetectionCritic   : object recall/precision via Hungarian-ish greedy  │
│                     label+IoU+centroid matching (src/critics/utils.py)│
│ CountCritic        : per-label count delta                            │
│ SegmentationCritic : mask IoU of matched objects                      │
│ SpatialCritic      : centroid displacement + area ratio               │
│ CLIPCritic         : CLIP(image_after, instruction)                   │
│ VLMCritic          : mock / GPT-4V / Qwen-VL / LLaVA hook (all        │
│                       non-mock branches currently call the mock)      │
│ PhysicsCritic      : rule table keyed on "{subj}_{relation}_{obj}"    │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
RewardAggregator.aggregate()  →  final_score = Σ wᵢ·scoreᵢ  (flat weighted sum)
        │
        ▼
PromptRefiner.refine()  →  refined_instruction (suggestions appended, never fed back)
        │
        ▼
results/{image_before,image_after,detections,masks,scene_graphs,scores,prompts}/*.json
```

This is a single-shot **evaluator**, not a closed loop: nothing calls an editor, nothing re-runs with the refined instruction. It's the scoring half of Direction 1, missing the generation/iteration half.

## 3. Bugs and suspicious logic

**CRITICAL-1 — System B is scientifically inert by default (`use_mock: true`).**
`src/configs/evaluation.yaml` ships with `use_mock: true`. In this mode: Grounding DINO returns `f(width, height, class_name)` — a lookup table of fixed box proportions — completely ignoring pixel content; SAM2 draws an ellipse inside that box; CLIPCritic returns the literal constant `0.78`; VLMCritic returns the literal constant `0.82` plus a templated string. Since before/after images are normally the same resolution, before/after scene graphs come out identical regardless of what the editor did, so detection/count/segmentation/spatial/physics all trivially score "perfect," and clip/vlm are hardcoded. **Any numbers produced by `run_evaluation.py` today, with default config, are not measuring the edit at all.** This needs to be the first thing fixed before System B is trusted for anything, including a demo.

**CRITICAL-2 — Composite/individual critic scores are computed twice per iteration (redundant model calls).**
In `ClosedLoopPipeline.run()`: `self.critic.score(...)` (the `CompositeCritic`) internally runs all 3 sub-critics once to get the weighted sum, discarding their individual values; then `self.critic.get_individual_scores(...)` is called immediately after, purely for logging/JSON, and **re-runs all 3 sub-critics from scratch** — a second full YOLO detection pass and two more CLIP forward passes, per iteration, purely to populate a log field. Over `max_iterations=3` × 15 images this silently doubles GPU/CPU cost for identical results. `CompositeCritic.score()` should return `(composite, individual)` in one pass, or `get_individual_scores` should be the only call and `score` should derive from it.

**CRITICAL-3 (research validity) — `ObjectConsistencyCritic` treats "same class detected" as "object preserved," with no positional or identity check beyond IoU of a single top-1 box.** If YOLO detects *a different person* who wandered into the same region of an unrelated background, or if the pose is destroyed but a low-confidence box still happens to overlap, this scores well. There's no re-identification signal (no CLIP region embedding, no keypoint/pose check) — just "is there a person-shaped box roughly where there used to be one."

**MAJOR-1 — Original image gets re-detected every iteration even though it never changes.** `self.detector.detect(original_image)` is called inside `ObjectConsistencyCritic.score` (×2, per CRITICAL-2) and *again* directly in `ClosedLoopPipeline.run()` for the visualization/detections dict — that's up to 3 YOLO passes on the same unchanging image per iteration. Should be detected once outside the loop and reused.

**MAJOR-2 — `main.py`, `src/closed_loop_editor.py`, and `experiments/run_single.py` are ~90% duplicated boilerplate.** All three: load config → `build_pipeline()` → load/resize image → `build_prompt()` → `pipeline.run()` → pick best iteration → write image/CSV. `closed_loop_editor.py`'s docstring literally says "delegates to the modular framework while preserving legacy behavior" — but it re-implements the loop rather than calling `main.py` with different defaults. These should be one script with a `--mode {batch,single,legacy}` or, more simply, `closed_loop_editor.py` should shell out to `main.py`'s functions instead of re-implementing the same 90 lines.

**MAJOR-3 — Two unrelated, differently-named prompt-refinement modules with overlapping purpose.** `src/prompts/refinement.py::refine_prompt` (used by System A, keys on `object_consistency`/`clip_similarity`/`instruction_alignment`) and `src/prompt_refinement/refiner.py::PromptRefiner.refine` (used by System B, keys on `detection`/`count`/`segmentation`/`spatial`/`clip`/`vlm`/`physics`). Same concept, same append-a-clause strategy, incompatible score-name vocabularies, in two different packages (`prompts/` vs `prompt_refinement/`) that a newcomer would assume are the same thing. One of these needs to absorb the other once System A/B are unified.

**MAJOR-4 — `bbox_iou` / `_bbox_iou` is implemented identically three times** (`src/detectors/yolo_detector.py::bbox_iou`, `src/critics/object_consistency.py::ObjectConsistencyCritic._bbox_iou`, `src/critics/utils.py::_bbox_iou`), byte-for-byte the same algorithm. Trivial to consolidate into `src/utils/geometry.py`, but worth doing before any of the three copies drifts and silently disagrees with the others.

**MAJOR-5 — Weighted-sum aggregation is implemented three separate times**, two of them live (`CompositeCritic.score`, `RewardAggregator.aggregate`) and one dead (`src/utils/io_utils.py::combine_metrics`, defined, never imported anywhere — confirmed by repo-wide grep). None of the three know about each other or share validation logic (weight-sum checks differ: `CompositeCritic` raises on mismatch, `RewardAggregator` warns and silently renormalizes, `combine_metrics` raises).

**MAJOR-6 — `get_clip_model()` cache keys on device only, not on `model_id`.** `src/critics/clip_utils.py` has a module-level `_clip_bundle` cache that's reused as long as the requested `device` matches, regardless of whether a different `model_id` was requested. If two critics in the same process ever request different CLIP checkpoints, the second one silently gets scored with the first one's weights. Currently harmless because every caller passes the same default `model_id`, but it's a latent bug the moment someone parameterizes it (e.g. an ablation over CLIP variants).

**MINOR-1 — `SpatialCritic` infers the image diagonal from `masks_before[0].shape`, defaulting to a hardcoded `384×384` if `masks_before` is empty**, rather than taking image dimensions as an explicit argument. Silently wrong displacement normalization if the pipeline is ever run at a resolution other than 384 with no detections in the first frame.

**MINOR-2 — `PhysicsCritic`'s rule matching is substring containment (`if key in rule_key or rule_key in key`), not exact match.** With more rules added later this is a real foot-gun: e.g. a rule `"car_floating_on_river"` would also match a constructed key like `"race_car_floating_on_river_bank"` or partially overlap with a shorter rule name added later. Fine for the current 4-rule table, fragile as the rule set grows — worth switching to exact dict lookup before Phase 5/6 experiments add more rules.

**MINOR-3 — `SegmentationCritic`/`SpatialCritic` recover a list index by string-parsing the scene-graph object id (`int(obj["id"].split("_")[1])`)** rather than the scene graph carrying an explicit `mask_index` field or the mask itself. It happens to be correct today because `SceneGraphBuilder` builds ids and truncates the mask list in lockstep, but it's an implicit invariant enforced only by convention across two files — a natural target for Phase 2's data-structure redesign (attach mask references directly to scene-graph nodes).

**DESIGN QUESTION-1 — `CountCritic`'s normalization is asymmetric and can exceed [0,1] internally before clamping.** `score = 1 - total_diff / len(objs_bef)` — if many more objects appear after editing than existed before (`objs_aft` count irrelevant to the denominator), the score can go very negative before the `max(0.0, ...)` clamp, meaning "5 hallucinated extra objects" and "50 hallucinated extra objects" both floor to 0.0 and become indistinguishable. Given your brief's interest in "unwanted-object detection," you may want this to be its own signal rather than folded into a symmetric count-diff.

**DESIGN QUESTION-2 — `DetectionCritic` returns raw recall as the score, with precision computed but not used in the score itself.** That means an edit that duplicates every original object 5× (spurious detections, precision tanks) scores identically to a clean 1:1 preservation, as long as recall is 1.0. Worth deciding whether "unwanted object" is detection's job (precision matters) or another critic's job.

---

## 4. Research-level critique: MODEL OUTPUT vs. RESEARCH METRIC vs. INTERPRETATION

| Critic | MODEL OUTPUT (what the model literally emits) | RESEARCH METRIC (what the code computes from it) | INTERPRETATION being asserted (what the pipeline treats it as meaning) | Gap |
|---|---|---|---|---|
| Grounding DINO / YOLO detection | Class label + confidence + box, per query term | Box IoU (System A) or Hungarian-style recall/precision over matched labels (System B) | "The subject was preserved / the count is correct" | Confidence and box overlap say nothing about *identity* — a different instance of the same class in the same region reads as "preserved." No re-ID signal exists anywhere in the repo. |
| SAM2 segmentation | Per-box binary mask | Mask IoU of matched objects | "Structural/boundary preservation" | A legitimate edit that changes background *behind* the subject's silhouette can lower IoU near boundaries for reasons unrelated to subject damage (anti-aliasing, shadow removal); the critic has no notion of "expected boundary change region," so it can't distinguish acceptable boundary drift from real damage — this is the exact concern you raised in the brief. |
| CLIP image↔image (`clip_similarity`) | Cosine similarity of global CLIP embeddings | Linear rescale to [0,1] | "Semantic preservation of the subject" | Global embedding conflates subject and background; a background-only edit and a subject-destroying edit that happens to preserve overall composition/colors can land at similar CLIP similarity. This is well documented in the CLIP-as-a-metric literature and is exactly the ambiguity your brief calls out ("high CLIP similarity necessarily mean correctly followed?" — no). |
| CLIP image↔text (`instruction_alignment` / `clip_alignment`) | Cosine similarity of image embedding to instruction text embedding | Linear rescale to [0,1] | "Instruction was followed" | CLIP text-image similarity is known to saturate/plateau and is weakly sensitive to *which* background appeared, only that *some* plausible scene matching keywords in the prompt appeared. It cannot distinguish "removed background cleanly" from "background replaced with something merely thematically related." |
| VLM critic | A single scalar + freeform reasoning text (currently: **hardcoded constant + templated string**, not a model output at all) | Passed straight through as `vlm` score | "An independent semantic/plausibility judge" | Right now there is no real VLM in the loop — it's a stub. Once wired to a real VLM, note: a single VLM call's score has no established reproducibility/calibration in this codebase (no repeat-sampling, no rubric, no calibration set) — your brief's own question ("how reproducible is this score?") is currently unanswerable because nothing real runs yet. |
| Physics critic | Nothing — this is 100% rule-based (`rewards`/`penalties` dict keyed on scene-graph relation strings) | A hand-authored lookup table (4 entries) added/subtracted from a 0.5 base | "Physical plausibility" | This is geometric/heuristic plausibility (bounding-box adjacency, not force/support/contact reasoning), and it's scoped to exactly the 7 hardcoded classes in `evaluation.yaml` (`person, dog, river, mountain, boat, fish, car`) — it cannot generalize to an unseen object without a new rule. Calling this "physics" is optimistic; "geometric plausibility heuristic" is the accurate name, matching your own caution in the brief about not overclaiming a physics simulator. |
| Scene graph relations (`standing_on`/`floating_on`/`inside`/`near`) | — | Derived purely from 2D box/mask geometry, no depth | "Support / containment relationships" | With no depth estimate, "near" and "standing_on" are indistinguishable from coincidental 2D overlap caused by camera angle (an object *behind* another at different depth can appear "standing_on" it in 2D). The repo currently has no depth model anywhere. |

---

## 5. Where the implementation diverges from your stated research design

- **Instruction-conditioned evaluation (your Creative Idea #1): not implemented anywhere.** Both systems apply the same fixed critic weights regardless of what the instruction asks for. `target_classes` in `evaluation.yaml` is a fixed global list ("person, dog, river, mountain, boat, fish, car"), not derived from the instruction per-example.
- **Critical-object graph / primary vs. secondary vs. background objects (Creative Idea #2): not implemented.** `DetectionCritic`/`CountCritic` treat every matched label symmetrically; there's no primary-subject flag anywhere in the scene graph schema.
- **Catastrophic failure gating (Creative Idea #3, and your own stated intuition about hard constraints): not implemented.** Both `CompositeCritic` and `RewardAggregator` are pure linear weighted sums — nothing in the code currently prevents a very high `clip`/`vlm` score from masking a `detection` score of 0.0 (a fully destroyed subject). This is worth prioritizing in Phase 2/5 since it's the one architectural question you flagged as a real hypothesis to test (Experiment 4 in your brief already asks for this comparison).
- **Counterfactual / edit-locality critic (Creative Ideas #4–5): not implemented.** No component currently asks "what changed that wasn't requested," and no component checks whether pixel changes were spatially concentrated where the instruction implies (background region) vs. bleeding into the foreground.
- **Iterative closed loop for System B: not implemented.** `PromptRefiner.refine()` computes a `refined_instruction` and writes it to `results/.../prompts/*.json`, but nothing re-invokes an editor with it. System A *does* close the loop, but only with 3 shallow critics, no scene graph, no physics/spatial reasoning.
- **Multi-object handling: partially blocking.** System A's `YOLODetector.detect()` is single-object by construction (returns one box). Any research question involving "did we keep the chair but not the table" (your object-matching example) cannot currently be asked in System A at all — it requires System B's multi-object detection + matching, which is the disconnected half.

---

## 6. What Phase 2 needs to decide before any code moves

1. **Commit to one integrated pipeline.** The two systems need to become one: System B's detector/segmenter/critics/scene-graph should replace System A's YOLO-only, single-critic loop inside `ClosedLoopPipeline`, so multi-object detection, scene graphs, and structured feedback actually drive the iterative loop instead of sitting in a side script.
2. **Fix CRITICAL-1 before trusting any System-B number** — either wire real Grounding DINO/SAM2/CLIP/VLM by default, or make it loud and impossible to miss when mock mode is active (e.g. tag every result JSON with `"mock": true` at the top level — currently nothing in the output artifacts flags this).
3. **Decide the aggregation question you already raised**: replace the flat weighted sum with your favored hybrid (hard-constraint gate on catastrophic subject loss, then weighted/learned combination for the rest) — this is Phase 2/5 architecture work, not a Phase 1 fix, but the audit confirms nothing about the current linear-sum implementation would need to survive that redesign; it's cleanly isolated in two small files (`composite.py`, `aggregator.py`).
4. **Decide what "physics critic" should honestly be called and scoped to** before it grows — a 7-class, 4-rule hardcoded lookup table won't survive contact with a broader dataset.

This concludes Phase 1. No source files were modified. Ready for Phase 2 (proposed architecture) on your go-ahead — in particular I'd like your call on point 1 above (integrate vs. keep separate) since it changes the shape of everything downstream.
