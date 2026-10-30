"""E2 step 3 — agreement between annotators, human verdict on the editors, and critic-vs-human agreement.

    python experiments/e2_analyze.py                                  # all annotators in annotation/e2/labels
    python experiments/e2_analyze.py --pilot                          # the calibration round
    python experiments/e2_analyze.py --map comp_v2=rescored_comp      # score with re-scored runs (critics v3)

Writes annotation/e2/report[_pilot].md and prints the same text.

Sections
 1. inter-annotator agreement: Cohen's kappa per yes/no question, quadratic-weighted kappa for the 1-5 rating,
    pairwise agreement; intra-annotator consistency on repeated items
 2. human verdict: yes-rate and mean rating per editor; comp vs qwen preference with a bootstrap CI that
    resamples SOURCE PHOTOS (the 90 tasks are 15 photos x 6 targets, not 90 independent samples)
 3. critics vs humans: pairwise accuracy of every score against the human preference, Spearman with the
    mean rating (all images and within each editor), AUROC of every critic for every yes/no question
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YN = ["bg", "keep", "support", "shadow", "light", "edge"]
PREF = ["pref_overall", "pref_keep", "pref_physics"]
QTEXT = {"bg": "背景换对", "keep": "主体保留", "support": "支撑", "shadow": "接触阴影", "light": "光照一致",
         "edge": "边缘干净", "overall": "整体 1-5", "pref_overall": "整体偏好", "pref_keep": "保留偏好",
         "pref_physics": "物理偏好"}


# ---------------------------------------------------------------- statistics helpers
def kappa(a: list, b: list, cats: list, weights: str | None = None) -> float | None:
    n = len(a)
    if n < 2:
        return None
    k = len(cats)
    ix = {c: i for i, c in enumerate(cats)}
    w = [[(1.0 if i != j else 0.0) if weights is None else ((i - j) / (k - 1)) ** 2 for j in range(k)]
         for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[ix[x]][ix[y]] += 1 / n
    pa = [sum(obs[i]) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return None if de == 0 else 1 - do / de


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else num / den


def auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    r = ranks(scores)
    rp = sum(ri for ri, l in zip(r, labels) if l == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fmt(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


# ---------------------------------------------------------------- loading
def load_labels(d: Path, pilot: bool) -> dict[str, dict[str, dict]]:
    out = {}
    for f in sorted((d / "labels").glob("*.jsonl")):
        if f.stem.endswith("_pilot") != pilot:
            continue
        name = f.stem.removesuffix("_pilot")
        recs = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r["answers"]
        out[name] = recs
    return out


def load_scores(runs_dir: Path, run: str, task: str, cache: dict) -> dict:
    k = (run, task)
    if k not in cache:
        rep = json.loads((runs_dir / run / task / "report.json").read_text())
        ev = rep.get("best_evaluation", rep)
        flat = {"overall": ev["overall"], **{f"branch:{b}": v for b, v in ev["branch_scores"].items()}}
        for c, v in ev.get("critics", {}).items():
            if v.get("score") is not None:
                flat[c] = v["score"]
        b = ev["branch_scores"]
        flat["follow×world"] = (b["follow"] * b["world"]) ** 0.5
        cache[k] = flat
    return cache[k]


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "annotation" / "e2"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--map", action="append", default=[], metavar="RUN=OTHER",
                    help="read critic scores of RUN from runs/OTHER (e.g. after re-scoring)")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="dev = source photos *_01/*_02, test = *_03 (tune critics on dev, report on test)")
    args = ap.parse_args()

    d, runs_dir = Path(args.dir), Path(args.runs_dir)
    key = json.loads((d / "key.json").read_text())
    K = key["items"]
    if args.split != "all":
        is_test = lambda k: k["task"].split("__")[0].endswith("_03")  # noqa: E731
        K = {i: k for i, k in K.items() if is_test(k) == (args.split == "test")}
    run_map = dict(m.split("=", 1) for m in args.map)
    L = load_labels(d, args.pilot)
    if not L:
        raise SystemExit("no labels found")
    names = sorted(L)
    lines: list[str] = []
    P = lines.append
    P(f"# E2 结果{'（pilot）' if args.pilot else ''}{'' if args.split == 'all' else f'（{args.split} 划分）'}\n")
    P(f"标注人：{', '.join(f'{n}（{len(L[n])} 条）' for n in names)}。critic 分数来源："
      f"{', '.join(f'{a}→{b}' for a, b in run_map.items()) or '原始 run'}。\n")

    # ---- 1. agreement
    P("## 1. 标注一致性\n")
    if len(names) >= 2:
        P("| 题目 | 两人都标的条数 | 一致率 | Cohen's κ |\n|---|---|---|---|")
        a, b = names[0], names[1]
        common = [i for i in L[a] if i in L[b] and i in K and "repeat_of" not in K[i]]
        for q in YN + ["overall"] + PREF:
            xs = [(L[a][i][q], L[b][i][q]) for i in common if q in L[a][i] and q in L[b][i]]
            if q in YN:
                xs = [x for x in xs if "3" not in x]  # drop "unsure"
                kap = kappa([x for x, _ in xs], [y for _, y in xs], ["1", "2"])
            elif q == "overall":
                kap = kappa([x for x, _ in xs], [y for _, y in xs], ["1", "2", "3", "4", "5"], weights="quadratic")
            else:
                kap = kappa([x for x, _ in xs], [y for _, y in xs], ["1", "2", "3"])
            agree = sum(x == y for x, y in xs) / len(xs) if xs else None
            P(f"| {QTEXT[q]}{'（加权 κ）' if q == 'overall' else ''} | {len(xs)} | {fmt(agree)} | {fmt(kap)} |")
        P("\nκ < 0.4 的题目说明定义不清，需要回到标注规范讨论（pilot 阶段）或在分析中降权。\n")
    for n in names:
        reps = [(i, K[i]["repeat_of"]) for i in L[n] if i in K and "repeat_of" in K[i] and K[i]["repeat_of"] in L[n]]
        if reps:
            same = [L[n][i][q] == L[n][o][q] for i, o in reps for q in YN]
            dr = [abs(int(L[n][i]["overall"]) - int(L[n][o]["overall"])) for i, o in reps]
            P(f"- {n} 自身一致性（{len(reps)} 个重复项）：是/否题一致 {fmt(sum(same) / len(same))}，"
              f"整体评分平均差 {fmt(st.mean(dr))}")
    P("")

    # ---- consensus per item
    def yn_val(v):
        return {"1": 1.0, "2": 0.0}.get(v)

    cons: dict[str, dict] = {}
    for i, k in K.items():
        if "repeat_of" in k:
            continue
        got = [L[n][i] for n in names if i in L[n]]
        if not got:
            continue
        c = {}
        if i.startswith("s"):
            for q in YN:
                vals = [yn_val(g[q]) for g in got if yn_val(g[q]) is not None]
                c[q] = st.mean(vals) if vals else None
            c["overall"] = st.mean(int(g["overall"]) for g in got)
        else:
            for q in PREF:
                # +1 = the editor shown as A wins, -1 = B wins, 0 = same
                c[q] = st.mean({"1": 1, "2": -1, "3": 0}[g[q]] for g in got)
        cons[i] = c

    singles = {i: c for i, c in cons.items() if i.startswith("s")}
    pairs = {i: c for i, c in cons.items() if i.startswith("p")}
    cache: dict = {}

    def scores_of(run, task):
        return load_scores(runs_dir, run_map.get(run, run), task, cache)

    # ---- 2. human verdict on editors
    P("## 2. 人对编辑器的判断\n")
    by_run = collections.defaultdict(list)
    for i, c in singles.items():
        by_run[K[i]["run"]].append(c)
    P("| 编辑器 | n | " + " | ".join(QTEXT[q] + "（是的比例）" for q in YN) + " | 整体 1-5 均值 |")
    P("|---" * (len(YN) + 3) + "|")
    for r, cs in sorted(by_run.items()):
        row = [fmt(st.mean([c[q] for c in cs if c[q] is not None])) for q in YN]
        P(f"| {r} | {len(cs)} | " + " | ".join(row) + f" | {fmt(st.mean(c['overall'] for c in cs))} |")
    P("")
    if pairs:
        x_run, y_run = key["pair_runs"]
        P(f"**{y_run} vs {x_run} 成对偏好**（正值 = {y_run} 更好；CI = 按源图做 cluster bootstrap 的 95% 区间）\n")
        P("| 题目 | " + f"{y_run} 胜 | {x_run} 胜 | 平 | {y_run} 胜率（不计平） | 95% CI |\n|---|---|---|---|---|---|")
        rng = random.Random(0)
        for q in PREF:
            per = []  # (source sample, value from the y_run perspective)
            for i, c in pairs.items():
                v = c[q] if K[i]["A"] == y_run else -c[q]
                per.append((K[i]["task"].split("__")[0], v))
            w = sum(v > 0 for _, v in per)
            l = sum(v < 0 for _, v in per)
            t = len(per) - w - l
            rate = w / (w + l) if w + l else None
            groups = collections.defaultdict(list)
            for s, v in per:
                groups[s].append(v)
            keys_ = list(groups)
            boots = []
            for _ in range(args.boot):
                vs = [v for s in (rng.choice(keys_) for _ in keys_) for v in groups[s]]
                ww, ll = sum(v > 0 for v in vs), sum(v < 0 for v in vs)
                if ww + ll:
                    boots.append(ww / (ww + ll))
            boots.sort()
            ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
            P(f"| {QTEXT[q]} | {w} | {l} | {t} | {fmt(rate)} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
        P("\n判定：整体偏好的 CI 下界 > 0.5 → 人更喜欢 " + y_run + "；上界 < 0.5 → 人更喜欢 " + x_run + "；跨过 0.5 → 无明显差别。\n")

    # ---- 3. critics vs humans
    P("## 3. critic 与人的一致性\n")
    if pairs:
        x_run, y_run = key["pair_runs"]
        names_s = ["overall", "branch:keep", "branch:follow", "branch:world", "branch:weighted_sum_baseline",
                   "follow×world"]
        P(f"**成对准确率**：在人有明确偏好（不平）的 pair 上，分数高的那张是否就是人选的那张。\n")
        P("| 分数 | " + " | ".join(QTEXT[q] for q in PREF) + " |\n|---" * 1 + "|---" * len(PREF) + "|")
        for s in names_s:
            row = []
            for q in PREF:
                hit = n = 0
                for i, c in pairs.items():
                    if c[q] == 0:
                        continue
                    t = K[i]["task"]
                    sa, sb = scores_of(K[i]["A"], t).get(s), scores_of(K[i]["B"], t).get(s)
                    if sa is None or sb is None or sa == sb:
                        continue
                    n += 1
                    hit += (sa > sb) == (c[q] > 0)
                row.append(f"{fmt(hit / n if n else None)} (n={n})")
            P(f"| {s} | " + " | ".join(row) + " |")
        P("")

    rows = []
    for i, c in singles.items():
        rows.append((K[i]["run"], scores_of(K[i]["run"], K[i]["task"]), c))
    all_keys = sorted({k for _, s, _ in rows for k in s}, key=lambda k: (k != "overall", not k.startswith("branch"), k))
    P("**Spearman（critic 分数 vs 人的整体评分）**\n")
    P("| 分数 | 全部 | " + " | ".join(sorted(by_run)) + " |\n|---|---|" + "---|" * len(by_run))
    for k in all_keys:
        cells = [fmt(spearman([s[k] for _, s, _ in rows if k in s], [c["overall"] for _, s, c in rows if k in s]))]
        for r in sorted(by_run):
            sub = [(s[k], c["overall"]) for rr, s, c in rows if rr == r and k in s]
            cells.append(fmt(spearman([a for a, _ in sub], [b for _, b in sub])))
        P(f"| {k} | " + " | ".join(cells) + " |")
    P("\n编辑器内部的 Spearman 更能说明问题：全部图混在一起时，只要能区分 ip2p 和其他两个编辑器就会很高。\n")

    P("**AUROC（critic 分数能否区分人判为「是」和「否」的图）**——只用两人一致的条目；> 0.7 算有用，≈ 0.5 等于没看出来。\n")
    P("| critic | " + " | ".join(QTEXT[q] for q in YN) + " | 整体 ≥ 4 |\n|---|" + "---|" * (len(YN) + 1))
    for k in all_keys:
        cells = []
        for q in YN:
            sub = [(s[k], int(c[q])) for _, s, c in rows if k in s and c[q] in (0.0, 1.0)]
            cells.append(fmt(auroc([a for a, _ in sub], [b for _, b in sub])))
        sub = [(s[k], int(c["overall"] >= 4)) for _, s, c in rows if k in s]
        cells.append(fmt(auroc([a for a, _ in sub], [b for _, b in sub])))
        P(f"| {k} | " + " | ".join(cells) + " |")
    P("")

    text = "\n".join(lines)
    out = d / f"report{'_pilot' if args.pilot else ''}{'' if args.split == 'all' else '_' + args.split}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
