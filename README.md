# Self-Improving Background Editor

Closed-loop **background replacement** that keeps the subject (person / dog) unchanged **and** keeps it
physically coupled to the new scene (support, contact shadow, lighting, clean edges).

```
before ─► EditSpec ─► perception (cached) ─┐
                                           ▼
   ┌── editor (N seeds) ─► after ─► perception ─► regions S/B/C/G ─► critics ─► gated aggregation ──┐
   │                                                                                                │
   └──────── router: issue type ─► action (prompt clause / parameter nudge / reseed) ◄──────────────┘
                      ▲ cross-sample action memory (UCB on Δscore)
```

Design docs: [`docs/specs/2026-09-30-pipeline-technical-design.md`](docs/specs/2026-09-30-pipeline-technical-design.md)
(pipeline) and [`docs/specs/2026-09-30-bg-replacement-critic-novelty-and-plan.md`](docs/specs/2026-09-30-bg-replacement-critic-novelty-and-plan.md)
(related work, contributions A–E, plan).

## Critics

| branch | critic | what it measures | region |
|---|---|---|---|
| **gate** | `subject_count` | exactly the original number of subjects (no missing / hallucinated extra) | — |
| | `identity` | DINOv2 CLS cosine of masked subject crops | S |
| | `bg_changed` | the background really changed (blocks the "return the input" reward hack) | G |
| **keep** | `silhouette` | mask IoU after similarity alignment | S |
| | `appearance` | Lab distance **after removing global lighting / colour cast** (lighting may change, texture may not) | S |
| | `texture` | DINOv2 patch cosine on the subject | S |
| **follow** | `bg_semantic` | SigLIP zero-shot: target vs all other backgrounds, subject removed | G |
| | `old_bg_residue` | source-scene concepts (e.g. water) left in the background | G |
| **world** | `support` | depth continuity between the subject's lowest pixels and the ground below | C |
| | `contact_shadow` | ground under the feet darker than its surroundings, relative to the real photo | C |
| | `light_harmony` | subject highlights share the scene illuminant colour, relative to the real photo | S, G |
| | `halo` | ring around the subject still looks like the old scene | B |

Aggregation (`aggregation.method`): `gated_geometric` (any gate fails → 0, else geometric mean of
keep/follow/world) or `weighted_sum` (legacy baseline, always logged as `weighted_sum_baseline`).

## Setup (Windows / Linux, 8 GB GPU is enough)

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
pip install -r requirements.txt
python -m pytest -q                                   # 21 CPU tests, no weights needed
python experiments/download_models.py                 # ~13 GB into the Hugging Face cache
```

## Run

```bash
# 0. if anything runs out of memory: per-model GPU/RAM usage report
python experiments/check_env.py

# 1. downscale originals + cache before-perception; check runs/prepare/*_regions.jpg
python experiments/prepare_data.py

# 2. smoke test: one image, one target
python experiments/run_loop.py --name smoke --samples dog_sit_river_01 --targets snow --save-candidates

# 3. full runs (15 images x 6 targets = 90 tasks)
python experiments/run_loop.py --name ip2p
python experiments/run_loop.py --name comp   --set loop.editor=compositing
python experiments/run_loop.py --name oneshot --set loop.max_edits=1 --set loop.n_candidates=1

# 4. E1 — do the critics see physical violations? (per-critic AUROC)
python experiments/critic_auroc.py --name e1

# score any existing pair
python experiments/evaluate_pair.py --before a.jpg --after b.jpg --object dog --action sit --background river --target snow
```

Every task writes `runs/<name>/<task>/{before.png, best.png, panel_best.jpg, report.json}`;
`runs/<name>/summary.csv` has one row per task. Any config key can be overridden with `--set a.b=value`.

On CHTC: `condor_submit chtc/submit.sub` (one GPU job per sample, `configs/chtc.yaml` overrides).

## Layout

```
configs/        default.yaml (8 GB laptop), chtc.yaml (overrides)
src/spec/       EditSpec: subject, source/target background, invariants & covariants
src/data/       labels.csv -> tasks, image loading (EXIF, aspect-preserving resize, cache)
src/models/     ModelRegistry (load once; per-model GPU policy keep / swap / unload), loaders
src/perception/ Grounding DINO, SAM 2, DINOv2, SigLIP, Depth Anything V2 -> Perception
src/regions/    S / B / C / G partition from masks
src/critics/    gate.py keep.py follow.py world.py
src/scoring/    aggregators
src/editors/    InstructPix2Pix, compositing baseline (SDXL inpainting)
src/refinement/ prompt clauses, issue -> action router, action memory
src/pipelines/  Evaluator, RefinementLoop
validation/     controlled physical-violation suite
experiments/    CLIs
tests/          CPU tests with synthetic scenes and fake models
```

`data/labels.csv` columns: `filename, object, action, background, submerged` (`submerged=1` when the
subject is partly in water in the original photo — the hardest case for covariant physics).
