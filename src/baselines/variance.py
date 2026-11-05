"""How stable are the VLM region-QA probabilities?

The VLM critics read P(Yes) from next-token logits, so repeating the SAME input gives the same number (up to GPU
nondeterminism) and "sampling 3 times" would measure nothing. What can vary, and what a reviewer will ask about, is the
input: the wording of the question, the crop padding, the image resolution. ``VIEWS`` perturb exactly those; ``orig_rep``
repeats the original view to measure the nondeterminism floor.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict

VIEWS = ("orig", "orig_rep", "para1", "para2", "pad_small", "pad_large", "side512")
PERTURBED = tuple(v for v in VIEWS if v != "orig_rep")


def view_params(view: str) -> dict:
    """Which knobs a view changes: paraphrase index (0 = original wording), crop padding, max image side."""
    base = {"para": 0, "pad": 0.15, "max_side": 768}
    return {"para1": {**base, "para": 1}, "para2": {**base, "para": 2}, "pad_small": {**base, "pad": 0.05},
            "pad_large": {**base, "pad": 0.30}, "side512": {**base, "max_side": 512}}.get(view, base)


def paraphrases(spec) -> dict[str, list[str]]:
    """question key -> [original, paraphrase 1, paraphrase 2]; 'Yes' always means physically / semantically fine."""
    from src.perception.vlm import build_questions

    s, pose, bg = spec.subject, spec.pose, spec.target_bg.name
    orig = {q.key: q.text for q in build_questions(spec)}
    para = {
        "support": [f"In the area inside the red box, is the {s} supported by something solid, and not hovering "
                    f"above it or sunk into it?",
                    f"Does the {s} appear to be properly in contact with the ground in the red box (neither "
                    f"floating nor sinking)?"],
        "shadow": [f"Inside the red box, is there a realistic shadow or darkening on the ground where the {s} "
                   f"meets it?",
                   f"Where the {s} touches the ground in the red box, does the ground look naturally shaded?"],
        "surface": [f"Does it make physical sense for the {s} to be {pose} on what is under it in this scene?",
                    f"Is the {s} {pose} on a believable surface, rather than on something impossible?"],
        "integration": [f"Does the {s} blend naturally into this photo, without cut-out edges, halos or outlines?",
                        f"Does the {s} look like it was really photographed in this scene, with no pasted-in look?"],
        "lighting": [f"Does the light on the {s} (direction, brightness, color) match the light in the rest of "
                     f"the scene?",
                     f"Is the {s} lit in the same way as its surroundings?"],
        "background": [f"Does this photo show a {bg} in the background?",
                       f"Is the scene behind the {s} a {bg}?"],
    }
    return {k: [orig[k], *para[k]] for k in orig}


def summarise(rows: list[dict]) -> dict:
    """rows: dicts with run, task, image, question, view, p_yes. Returns per-question spread and the repeat floor."""
    by = defaultdict(dict)  # (run, task, image, question) -> {view: p}
    for r in rows:
        if r.get("p_yes") is not None:
            by[(r["run"], r["task"], r["image"], r["question"])][r["view"]] = r["p_yes"]
    per_q = defaultdict(lambda: {"std": [], "range": [], "dev_orig": []})
    floor = []
    for (_, _, _, q), v in by.items():
        vals = [v[x] for x in PERTURBED if x in v]
        if len(vals) >= 2:
            per_q[q]["std"].append(st.pstdev(vals))
            per_q[q]["range"].append(max(vals) - min(vals))
        if "orig" in v:
            per_q[q]["dev_orig"] += [abs(v[x] - v["orig"]) for x in PERTURBED if x in v and x != "orig"]
            if "orig_rep" in v:
                floor.append(abs(v["orig_rep"] - v["orig"]))
    per_view = defaultdict(lambda: defaultdict(list))  # question -> view -> |p_view - p_orig|
    for (_, _, _, q), v in by.items():
        if "orig" in v:
            for x in PERTURBED:
                if x != "orig" and x in v:
                    per_view[q][x].append(abs(v[x] - v["orig"]))
    out = {"repeat_floor_max": max(floor) if floor else None, "repeat_floor_mean": st.mean(floor) if floor else None,
           "views": {q: {x: st.mean(d) for x, d in vs.items()} for q, vs in per_view.items()}, "questions": {}}
    for q, d in per_q.items():
        out["questions"][q] = {"n": len(d["std"]), "mean_std": st.mean(d["std"]) if d["std"] else None,
                               "mean_range": st.mean(d["range"]) if d["range"] else None,
                               "max_range": max(d["range"]) if d["range"] else None,
                               "mean_dev_from_orig": st.mean(d["dev_orig"]) if d["dev_orig"] else None}
    return out


def ranking_stability(rows: list[dict], run_a: str, run_b: str, image: str = "best") -> dict:
    """Does 'run_a beats run_b' on a question survive the perturbations? Per question and view: wins of a over b on the
    common tasks, plus the share of tasks whose sign never flips across the perturbed views."""
    p = {}
    for r in rows:
        if r["image"] == image and r.get("p_yes") is not None:
            p[(r["run"], r["task"], r["question"], r["view"])] = r["p_yes"]
    out = {}
    qs = sorted({k[2] for k in p})
    for q in qs:
        tasks = sorted({k[1] for k in p if k[0] == run_a and k[2] == q} & {k[1] for k in p if k[0] == run_b and k[2] == q})
        views = {}
        for v in VIEWS:
            d = [p[(run_a, t, q, v)] - p[(run_b, t, q, v)] for t in tasks
                 if (run_a, t, q, v) in p and (run_b, t, q, v) in p]
            if d:
                views[v] = {"a_wins": sum(x > 1e-9 for x in d), "b_wins": sum(x < -1e-9 for x in d),
                            "mean_diff": st.mean(d), "n": len(d)}
        stable = 0
        for t in tasks:
            signs = {(p[(run_a, t, q, v)] - p[(run_b, t, q, v)]) > 0 for v in PERTURBED
                     if (run_a, t, q, v) in p and (run_b, t, q, v) in p}
            stable += len(signs) == 1
        out[q] = {"views": views, "tasks": len(tasks), "sign_stable_tasks": stable}
    return out
