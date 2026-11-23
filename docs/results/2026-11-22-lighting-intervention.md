# Lighting-intervention test of the paper's light normalisation ℒ (item 1)

Run `lint1`, 2026-11-22, CHTC, 15 source photos, 28 variants each (420 scored images), no editor, no VLM.
Code: `validation/interventions.py`, `experiments/lighting_intervention.py` (commit 71f203d). Raw tables: `docs/results/lint1/` (summary.csv, by_kind.csv).

## Question
The paper (Sec. 3, Eq. 5 and 7) says identity and appearance are compared on ℒ(x), a per-channel standardisation inside the subject mask, so that a change of lighting is not counted as a change of the subject. The code never applies ℒ: identity uses the raw DINOv2 CLS cosine, appearance v3 uses high-pass ZNCC plus a chroma term. Does ℒ separate "lighting changed" from "subject changed"? Does it beat the raw measures?

## Setup
Every variant is made from the same real photo by a known edit inside the subject mask only.
- LIGHT (should not be penalised): brightness gain, side-light gradient, colour cast, combined relight; levels 1 to 3.
- STRUCT (should be penalised): blur, texture shuffle (same colours and shading, pattern moved), hue rotation, mirror; levels 1 to 3 (mirror level 3 only).
- null = zero-strength relight; nuisance = blur 0.8 + sub-pixel shift + JPEG 80 (what an editor's re-encoding does).
- Measures: silhouette, ksil (paper Eq. 7), appearance v3, appearance v2, texture, raw identity cosine, kapp_lab and kapp_rgb (paper's cos(φ(ℒx), φ(ℒx')), ℒ in Lab or RGB).

## Result against the pre-registered criteria: no measure passes, and the criteria turned out to be uninformative
The criteria were fixed in the script docstring before the run (P0 nuisance AUROC ≤ 0.65; P1 light ≤ 0.65; P2 struct ≥ 0.85; P3 light-vs-struct separation ≥ 0.85).

| measure | P0 nuisance | P1 light | P2 struct | P3 separation [95% CI] |
|---|---|---|---|---|
| silhouette | 1.00 | 0.99 | 1.00 | 0.83 [0.76, 0.90] |
| ksil (paper) | 1.00 | 1.00 | 0.99 | 0.55 [0.49, 0.61] |
| appearance v3 | 1.00 | 1.00 | 1.00 | 0.93 [0.89, 0.97] |
| appearance v2 | 1.00 | 1.00 | 1.00 | 0.97 [0.95, 0.99] |
| texture | 0.50 | 0.50 | 0.67 | 0.67 [0.62, 0.71] |
| identity (raw) | 1.00 | 1.00 | 1.00 | 0.97 [0.94, 0.99] |
| kapp_lab | 1.00 | 0.99 | 1.00 | 0.98 [0.96, 1.00] |
| kapp_rgb | 1.00 | 0.99 | 1.00 | 0.97 [0.94, 0.99] |

Why P0 to P2 say nothing: the null is the pixel-identical photo, so it scores at the ceiling and any variant, even the harmless nuisance, ranks below it (AUROC 1.00). A rank test against the null measures "is there any change", not "how much". Texture is the opposite case: its ramp clips at 1.0, so almost every variant ties with the null (0.50). Only P3 (light vs struct of the same photo) is informative, and there kapp_lab is nominally highest but its CI overlaps raw identity (0.98 vs 0.97). The criteria are not changed after the fact; the analysis below is post-hoc and labelled so.

## Post-hoc: size of the penalty (mean score drop vs null; not pre-registered)
| measure | light 1-2 | light 3 | nuisance | blur/shuffle 2-3 | hue 2-3 | mirror |
|---|---|---|---|---|---|---|
| silhouette | 0.007 | 0.009 | 0.030 | 0.013 | 0.008 | 0.405 |
| appearance v3 | 0.027 | 0.058 | 0.085 | 0.178 | 0.340 | 1.000 |
| appearance v2 | 0.066 | 0.136 | 0.220 | 0.226 | 0.522 | 0.669 |
| texture | 0.000 | 0.000 | 0.000 | 0.044 | 0.000 | 0.558 |
| identity (raw) | 0.012 | 0.026 | 0.039 | 0.219 | 0.070 | 0.050 |
| kapp_lab | 0.009 | 0.015 | 0.046 | 0.217 | 0.081 | 0.054 |
| kapp_rgb | 0.010 | 0.021 | 0.042 | 0.222 | 0.064 | 0.052 |

Share of (photo, variant) pairs whose drop is larger than that photo's nuisance drop (1.0 = always above the re-encoding noise):
| measure | light 1-2 | light 3 | blur/shuffle 2-3 | hue 2-3 | mirror |
|---|---|---|---|---|---|
| appearance v3 | 0.07 | 0.31 | 0.62 | 1.00 | 1.00 |
| identity (raw) | 0.04 | 0.31 | 1.00 | 0.73 | 0.60 |
| kapp_lab | 0.01 | 0.05 | 0.98 | 0.77 | 0.53 |

## What this shows
1. ℒ helps a little, and it does no harm. Against raw identity, ℒ in Lab cuts the penalty for strong lighting (level 3) from 0.026 to 0.015; the paired difference over the 15 photos is 0.011 [0.007, 0.017]. At level 1-2 it is 0.002 [0.0004, 0.005]. The penalty for blur and shuffle does not change (difference 0.002 [-0.015, 0.018]). RGB-ℒ gives a smaller gain (0.006 [0.002, 0.011] at level 3, not distinguishable from zero at level 1-2). These intervals are over 15 photos and descriptive; the comparison was not pre-registered.
2. The raw DINOv2 identity cosine is already almost invariant to the lighting edits (drop ≤ 0.034 at the strongest level). So the paper's claim that ℒ is what makes identity lighting-robust is only weakly supported: most of the robustness comes from the encoder.
3. The identity measures (raw or ℒ) are nearly blind to a mirrored subject (drop 0.05) and weak on hue change (0.06 to 0.10 at 60° to 120°). They catch texture destruction (blur, shuffle). Mirror is caught by silhouette (0.41), ksil (0.26), appearance v3 (1.00) and texture (0.56). So "ρ is conserved" cannot be carried by the identity gate alone.
4. Appearance v3 handles side light and brightness well (drops 0.007 to 0.05) and detects colour change and texture change, but it penalises a colour cast of level 3 by 0.13 (cast is lighting by definition). v2 penalises every lighting edit more (up to 0.20) and its nuisance drop (0.22) is as large as a strong structural edit, which is why v2 prefers copy-pasted subjects.
5. The re-encoding nuisance alone costs more than any lighting edit of level 1-2 for every measure (e.g. kapp_lab 0.046 vs 0.009). Any threshold on these measures has to sit above the nuisance level, so mild structural edits (shuffle of 4 px, blur σ 1.5) are not separable from re-encoding by appearance v3 or silhouette.
6. Texture is saturated (ramp 0.4 to 0.85 clips at 1.0): it sees only mirror, σ 6 blur and 20 px shuffle. It carries almost no information in the Keep branch as configured.
7. ksil (the paper's Eq. 7 silhouette, |M∩M'|/|M|) sees only mirror, and is no better than the IoU the code uses.

## Limits
- Synthetic Lab edits, uniform over the subject (except the side gradient). Real relighting by an editor also changes cast shadows and specular highlights; this test does not cover it.
- 15 photos, 3 per scene; dev (_01, _02) and test (_03) give the same picture (see summary.csv), so no split-specific conclusion.
- ℒ was tested as per-channel standardisation in Lab and RGB only; "per-channel standardization [confirm implementation]" in the paper does not say which space.
- The pass/fail criteria need replacing (see next step); nothing here is a claim about which editor is better.

## Next
- Replace the rank-vs-null criteria with magnitude criteria written before the next run: a measure's penalty for LIGHT level ≤ 2 must stay below its own nuisance penalty, and its penalty for STRUCT level ≥ 2 must be above it for every STRUCT kind.
- Rescale texture (ramp lo/hi) on dev only, using the nuisance and level-1 drops, then rerun on dev and report test once.
- Test appearance v3 with a chroma term that is blind to a global cast of any size, and a hue-sensitive identity descriptor, on the same variants.
- Run the same variants on real editor outputs (item 2 of the plan) to see whether the lighting levels here match what editors produce.
