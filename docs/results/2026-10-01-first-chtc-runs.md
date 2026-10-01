# First CHTC runs — 2026-09-30 / 10-01

Runs: `ip2p` (InstructPix2Pix, 15 images x 6 targets = 90 tasks), `comp` (compositing baseline: SAM mask + SDXL
inpainting, 90 tasks), `e1` (critic AUROC on the controlled violation suite, 15 images x 7 violations).
Config: `configs/chtc.yaml` (1024 px, 4 seeds/round, budget 12 edits, threshold 0.75). GPUs: L40 / H200.

## 1. Editors

| | ip2p | comp |
|---|---|---|
| gate passed (best of loop) | 79/90 | **90/90** |
| strict success (gate + SigLIP top-1 = target) | 52/90 | **87/90** |
| median overall score | 0.672 | **0.913** |
| round-0 gate pass → after loop | 43 → 79 | 90 → 90 |
| tasks improved by the loop | 62 (mean +0.35) | 4 (already above threshold at round 0) |
| median score, submerged subjects | 0.625 (dogs 0.32) | 0.868 |
| comp beats ip2p | | 86/90 |

* **IP2P rarely replaces the background**; it recolours / restyles the whole image ("winter" = cooler, "forest" =
  green tint). The two-sided gate catches this: 268 of 579 gate-zero candidates scored > 0.65 under the legacy
  weighted sum (contribution D). The loop's "edit_more" action (lower image guidance) is what rescues most tasks.
* **Compositing** always replaces the background and keeps the subject, so it passes every gate. Its failures are
  physical (objects hugging the mask, odd support), which is where the world critics must do the work.

## 2. E1 — violation suite AUROC (real photo vs one violation)

| critic | float | sink | scale↑ | scale↓ | no_shadow | color_cast | halo* |
|---|---|---|---|---|---|---|---|
| support | **0.93** | 0.85 | 0.75 | 0.82 | 0.89 | 0.82 | 0.42 |
| contact_shadow | 0.41 | 0.40 | 0.54 | 0.59 | **0.36** | 0.49 | 0.53 |
| light_harmony | 0.83 | 0.80 | 0.80 | 0.83 | 0.87 | **1.00** | 0.47 |
| halo | – | – | – | – | – | – | **0.93** |

\* halo is measured against a clean paste on the same foreign background.

* support detects floating, light_harmony detects colour casts, halo detects halos.
* **contact_shadow is blind / inverted (0.36 on no_shadow)** — the luminance heuristic does not work on real photos.
* **Confound:** positives are untouched photos, negatives go through clean-plate inpainting + paste, which leaves
  smudges and mask-hole speckles. Keep-branch critics therefore reach AUROC 1.0 trivially, and part of the world
  critics' signal may be processing artefacts. Fix: positive = same pipeline with a null violation (dx = 0).

## 3. Critic problems found

1. `appearance` too strict (flags drift in 88/90 ip2p and 48/90 comp tasks, including pixel-exact pastes) and it
   penalises *correct* covariant changes, e.g. legs that were under water and are now regenerated on rocks.
   → exclude regions not visible in the before image (old water mask ∩ subject), recalibrate sigma.
2. `bg_changed` only gates; the amount of change does not enter the score, so restyled-but-not-replaced images
   still score ~0.68. → feed it into the follow branch, raise the gate threshold to 0.2–0.25.
3. `contact_shadow` → replace by VLM region QA (and re-test on the fixed E1).
4. Missed failure types (need semantics): rocks turned into sofa cushions under a sitting person (ip2p, scored 0.90);
   inpainting hallucinating a tent-like object hugging the subject outline (comp, halo only 0.88).
5. Threshold 0.75 is below comp's round-0 median, so the loop never iterates for comp.
   → compare loops at equal full budget, or raise the threshold.
