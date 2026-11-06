<!-- Pasted by the user from CHTC: runs/vlmvar/variance_report.md (Qwen2.5-VL-7B, comp_v2 + qwen, 90 tasks each), job rw run `vlmvar`, 2026-11-05. -->
# VLM answer stability

Repeat floor (same input twice): mean 0.000, max 0.000

| question | n | mean std over views | mean range | max range | mean |view-orig| |
|---|---|---|---|---|---|
| background | 360 | 0.015 | 0.045 | 0.665 | 0.011 |
| integration | 360 | 0.163 | 0.508 | 0.798 | 0.146 |
| lighting | 360 | 0.089 | 0.251 | 0.655 | 0.074 |
| shadow | 360 | 0.177 | 0.496 | 0.817 | 0.181 |
| support | 360 | 0.079 | 0.220 | 0.527 | 0.081 |
| surface | 360 | 0.153 | 0.426 | 0.782 | 0.130 |

## Which perturbation moves the answer most? mean |p(view) - p(orig)|

| question | para1 | para2 | pad_small | pad_large | side512 |
|---|---|---|---|---|---|
| background | 0.028 | 0.010 | 0.000 | 0.000 | 0.019 |
| integration | 0.175 | 0.301 | 0.078 | 0.056 | 0.120 |
| lighting | 0.140 | 0.175 | 0.000 | 0.000 | 0.057 |
| shadow | 0.285 | 0.366 | 0.081 | 0.075 | 0.100 |
| support | 0.096 | 0.150 | 0.046 | 0.055 | 0.056 |
| surface | 0.384 | 0.139 | 0.037 | 0.035 | 0.057 |

## comp_v2 vs qwen on the best images: wins of comp_v2 per view (sign-stable = same winner in every perturbed view)

| question | tasks | sign-stable tasks | orig | para1 | para2 | pad_small | pad_large | side512 |
|---|---|---|---|---|---|---|---|---|
| background | 90 | 66 | 6/83 | 16/74 | 9/78 | 6/83 | 6/83 | 6/81 |
| integration | 90 | 18 | 32/57 | 47/43 | 34/56 | 41/49 | 34/56 | 36/53 |
| lighting | 90 | 44 | 25/65 | 31/58 | 39/51 | 25/65 | 25/65 | 25/63 |
| shadow | 90 | 32 | 47/43 | 54/36 | 45/45 | 50/40 | 48/42 | 50/40 |
| support | 90 | 47 | 27/62 | 28/62 | 24/66 | 23/65 | 26/63 | 22/68 |
| surface | 90 | 52 | 14/76 | 35/55 | 21/68 | 16/72 | 18/72 | 21/69 |
