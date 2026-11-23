# Lighting-intervention test, part 2: magnitude criteria and cast-blind colour terms (item 1, run lint2)

Run `lint2`, 2026-11-23, CHTC, same 15 photos and the same 28 variants per photo as lint1 (420 scored images). Code: `experiments/lighting_criteria.py`, `validation/interventions.py::chroma_terms` (commit 657bb9f). Raw table: `docs/results/lint2/criteria.csv`. Part 1: `2026-11-22-lighting-intervention.md`.

## What changed since part 1
Part 1 found that the rank-vs-null criteria say nothing (the null is the pixel-identical photo, so everything ranks below it). Part 2 asks the useful question: is a change larger than what an editor's re-encoding does to an unchanged subject? A variant is **beyond noise** for a measure when it scores below the same photo's nuisance variant (blur 0.8 + sub-pixel shift + JPEG 80). This is unchanged by any monotone rescaling, so no ramp is tuned.

Criteria (fixed in the script before the lint2 run): M1, a measure is eligible if at most 0.10 of the LIGHT level 1-2 variants are beyond noise; M2, it covers a STRUCT kind if at least 0.80 of that kind's level >= 2 variants are beyond noise; U1/U2, the eligible measures together flag at most 0.20 of the lighting variants and cover all four kinds (blur, shuffle, hue, mirror).

**Honesty note.** The thresholds were chosen after lint1 had been summarised, on the same photos and variants. Passing them here is a re-analysis, not a held-out result. What is new in lint2 are the extra measures (chroma_std, chroma_cov, appearance_v3c, raw texture_cos, app_zncc). Reruns are not bit-identical (identity_raw on mirror moved from 0.60 to 0.53 between lint1 and lint2), so differences of a few points are noise.

## Result (all 15 photos): share of variants beyond re-encoding noise
| measure | light 1-2 | light 3 | blur | shuffle | hue | mirror | eligible | covers |
|---|---|---|---|---|---|---|---|---|
| silhouette | 0.02 | 0.04 | 0.00 | 0.07 | 0.00 | 1.00 | yes | mirror |
| ksil (paper) | 0.15 | 0.21 | 0.07 | 0.20 | 0.23 | 1.00 | no | mirror |
| appearance v3 | 0.07 | 0.31 | 0.63 | 0.60 | 1.00 | 1.00 | yes | hue, mirror |
| appearance v2 | 0.04 | 0.21 | 0.97 | 0.00 | 1.00 | 1.00 | yes | blur, hue, mirror |
| texture (ramped) | 0.00 | 0.00 | 0.40 | 0.30 | 0.00 | 0.93 | yes | mirror |
| texture_cos (raw) | 0.33 | 0.76 | 1.00 | 1.00 | 0.97 | 1.00 | no | all four |
| identity (raw) | 0.04 | 0.31 | 1.00 | 1.00 | 0.73 | 0.53 | yes | blur, shuffle |
| kapp_lab | 0.01 | 0.05 | 0.97 | 1.00 | 0.77 | 0.53 | yes | blur, shuffle |
| kapp_rgb | 0.01 | 0.15 | 1.00 | 1.00 | 0.60 | 0.53 | yes | blur, shuffle |
| app_zncc (structure only) | 0.00 | 0.11 | 1.00 | 1.00 | 0.00 | 1.00 | yes | blur, shuffle, mirror |
| app_chroma (v3 colour term) | 0.07 | 0.31 | 0.57 | 0.00 | 1.00 | 1.00 | yes | hue, mirror |
| chroma_std (new) | 0.07 | 0.23 | 0.57 | 0.00 | 1.00 | 1.00 | yes | hue, mirror |
| chroma_cov (new) | 0.24 | 0.45 | 1.00 | 0.07 | 1.00 | 0.20 | no | blur, hue |
| appearance_v3c (new) | 0.07 | 0.23 | 0.63 | 0.37 | 1.00 | 1.00 | yes | hue, mirror |

All eligible measures together (U1/U2): false-flag on light 1-2 = 0.13 (limit 0.20), coverage blur, shuffle, hue, mirror = 1.00 each. Dev photos (_01, _02): 0.12 and all 1.00; test photos (_03): 0.15 and all 1.00. Both pass.

## What this shows
1. **No single measure carries "the subject did not change".** Every eligible measure misses at least one kind: silhouette and texture see only mirror, app_zncc is blind to hue (0.00), identity and kapp are weak on hue (0.60 to 0.77) and mirror (0.53), appearance v3 misses about 40% of blur and shuffle. The Keep branch needs several measures, each covering a different kind.
2. **A small union works.** kapp_lab + app_zncc + app_chroma: false-flag on light 1-2 = 0.07 [0.03, 0.12], coverage 1.00 on all four kinds. With raw identity in place of kapp_lab: 0.10 [0.05, 0.16], so ℒ again lowers the lighting false-flag. Choosing this combination was done on the same data and is exploratory.
3. **The new colour terms did not fix the colour problem.** chroma_std is beyond noise on 60% of the strongest colour casts (v3's chroma term: 93%), still far above 0.10, and its own re-encoding drop is large (0.27, versus 0.085 for v3's term), because standardising amplifies noise. chroma_cov is not eligible (0.24 on light 1-2) and does not see mirror (0.20). For a roughly single-coloured subject a strong cast and a hue rotation are nearly the same edit, so no colour term can accept one and reject the other there.
4. **Strong lighting is where measures still fail.** At level 3: gain, appearance v3 0.23, identity 0.30; strong cast, v3 0.93; strong combined relight, identity 0.33. kapp_lab (0.03 to 0.07) and app_zncc (0.0 to 0.2) stay almost clean; raw texture cosine does not (0.76).
5. **Raw texture cosine separates structure well (1.00 on blur, shuffle, mirror) but fires on 33% of mild lighting,** so it is not eligible as is. The ramped texture critic fixed that by going blind (0.00 to 0.40).

## Limits
- Same photos and variants as lint1; thresholds set after seeing lint1. Treat as a design check, not a validation.
- Synthetic Lab edits; real relighting changes cast shadows and highlights. A union of "any measure flags" is a detector, not the score the loop uses (gated geometric mean of Keep, Follow, World); how to turn this into a score is not tested.
- 15 photos, 3 per scene.

## Next
- Run these variants on real editor outputs (item 2 of the plan): are real lighting changes inside the level 1-2 range here?
- Decide the Keep design: replace the single identity gate by kapp_lab (ℒ in Lab) + app_zncc + a colour term, and test it on the violation suite v4 (does it keep the AUROCs, does it stop preferring copy-paste?).
- Fresh photos for a real held-out check (the 15 photos are now used up for this question).
