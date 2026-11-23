# Lighting test, part 3: real editor outputs on the synthetic ladder (item 1, run rl1)

Run `rl1`, 2026-11-23, CHTC. 270 real outputs (comp_v2, qwen, ip2p_v2: 15 photos x 6 target backgrounds each; before.png vs best.png) scored with the same 14 Keep-type measures as lint1/lint2, plus the 28-variant synthetic ladder of each source photo (420 ladder rows). Code: `experiments/real_lighting.py`, `lighting_criteria.py --real` (commit c3986ee). Table: `docs/results/rl1/bands.csv`. Parts 1 and 2: `results/2026-11-22-lighting-intervention.md`, `results/2026-11-23-lighting-magnitude.md`.

## Question
Part 2 left one gap: are the synthetic lighting edits the right size? Where do the changes real editors make fall on that scale?

## Method
Each real output is placed in one band per measure, against the SAME photo's ladder (no threshold tuned): **A** within re-encoding noise (score >= the photo's nuisance score); **B** worse than noise but not below the weakest strong-lighting (level 3) variant; **C** below every strong-lighting variant; **D** below the median of the structure (level >= 2) variants. A band says how big the change is, not whether it is lighting.

## Result: share of real outputs per band (A / B / C, D)
| measure | comp_v2 | qwen | ip2p_v2 |
|---|---|---|---|
| app_zncc (structure only) | 1.00 / 0 / 0 | 0.09 / 0.01 / 0.90, D 0.53 | 0.00 / 0.00 / 1.00, D 0.31 |
| app_chroma (colour term of v3) | 1.00 / 0 / 0 | 0.16 / 0.27 / 0.57, D 0.75 | 0.00 / 0.22 / 0.78, D 1.00 |
| appearance v3 | 1.00 / 0 / 0 | 0.16 / 0.19 / 0.65 | 0.00 / 0.21 / 0.79 |
| kapp_lab | 0.67 / 0.04 / 0.29 | 0.33 / 0.03 / 0.63 | 0.09 / 0.02 / 0.89 |
| identity (raw) | 0.67 / 0.10 / 0.23 | 0.20 / 0.17 / 0.63 | 0.04 / 0.09 / 0.87 |
| silhouette | 0.24 / 0.07 / 0.69 | 0.09 / 0 / 0.91 | 0.26 / 0.08 / 0.67 |
| texture_cos (raw) | 0.01 / 0.50 / 0.49 | 0.00 / 0.12 / 0.88 | 0.00 / 0.19 / 0.81 |
| texture (ramped) | 0.99 / 0 / 0.01 | 0.74 / 0 / 0.26 | 0.70 / 0 / 0.30 |

Share of tasks flagged (below the nuisance score on ANY measure of the set): app_zncc + app_chroma: comp_v2 0.00, qwen 0.92, ip2p_v2 1.00. kapp_lab + app_zncc + app_chroma: comp_v2 0.33, qwen 0.96, ip2p_v2 1.00.

## What this shows
1. **The question cannot be answered from these outputs: real editors change much more than lighting.** On the pixel-based measures, 78% to 100% of the qwen and ip2p_v2 outputs are in band C, below even the strongest synthetic lighting edit, and half or more of the zncc scores are below the median structure edit (qwen median zncc 0.71, 10th percentile 0.08). A regenerated subject has new fine structure, so lighting cannot be isolated, and the lighting range matters less in practice than structure.
2. **comp_v2 pastes the subject back unchanged, and the pixel measures say so exactly.** app_zncc is 1.000 for all 90 outputs; appearance v3, v2 and the colour terms are 100% band A. This is the known preference of Keep measures for copy-paste, not evidence that comp_v2 is a better editor.
3. **The synthetic ladder understates the noise of mask- and context-based measures.** On the unchanged comp_v2 subject: silhouette is beyond the lighting range in 69% (the new background changes the segmentation mask), kapp_lab and raw identity in 33% and 33% (the masked crop changes), raw texture cosine in 99% (DINOv2 patch features see the new background). In the ladder the background never changes, so these measures looked clean there. Their false-flag on real outputs is 0.33 to 0.99, not 0.01 to 0.07.
4. **The lint2 recommendation changes.** The union kapp_lab + app_zncc + app_chroma gave 0.07 false-flag on synthetic lighting but flags 33% of the untouched comp_v2 subjects; dropping kapp_lab gives 0.00 on comp_v2 while still flagging 92% (qwen) and 100% (ip2p_v2). The pixel-based pair (app_zncc for structure and mirror, app_chroma for hue) is the one that survives this check. Its known weaknesses are unchanged: app_chroma is beyond noise on 93% of strong casts (lint2).
5. **Median scores sit near the noise level for identity-type measures.** comp_v2's median identity 0.977 and kapp_lab 0.972 equal the ladder's nuisance medians (0.976, 0.963), so "below the nuisance score" is close to a coin flip for them; they have no headroom between "untouched" and "re-encoded".

## Limits
- No human labels: nothing here says an output is good or bad, only how far it is from the original subject.
- comp_v2 is assumed to paste the subject unchanged because zncc is exactly 1.000; its crops and masks still differ.
- One nuisance operator defines "noise"; real re-encoding by a diffusion model is stronger than blur + shift + JPEG, which would raise band A for the regenerating editors.
- Same 15 photos as lint1/lint2.

## Next
- Keep redesign candidate: pixel-based measures (app_zncc, app_chroma) in the Keep branch; identity and silhouette only as gates with thresholds set on real masks. Test it on the violation suite v4: AUROC kept? Still prefers copy-paste?
- Re-fit the silhouette and identity tolerances on real before/after masks (dev photos only), then report test once.
- Decide what the paper may claim: the light normalisation helps on synthetic lighting (parts 1 and 2), but real editors' differences are structural, and mask/context-based measures do not transfer from the synthetic ladder.
