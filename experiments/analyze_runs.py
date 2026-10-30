"""Summarise finished loop runs from their report.json files.

Usage:
    python experiments/analyze_runs.py ip2p_v2 comp_v2 qwen
    python experiments/analyze_runs.py ip2p_v2 comp_v2 qwen --runs-dir runs --csv

Reads runs/<name>/<task>/report.json (not summary.csv, which collect.sh may overwrite)
and prints: gate pass rates (one-shot / round-0 best-of-N / after loop), score
distributions, branch and per-critic medians, issue counts, per-action acceptance and
mean delta, breakdown by target and subject, pairwise wins, and reward-hacking counts
(gate = 0 but legacy weighted sum > 0.65). With --csv it also rebuilds summary.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path


def load(runs_dir: Path, name: str) -> dict[str, dict]:
    out = {}
    for f in sorted((runs_dir / name).glob("*/report.json")):
        out[f.parent.name] = json.loads(f.read_text())
    return out


def ms(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.median(xs):.3f} / {st.mean(xs):.3f}" if xs else "-"


def round0(r):
    return [c for c in r["candidates"] if c["round"] == 0]


def summarise(name: str, D: dict[str, dict]) -> None:
    print(f"\n===== {name}  ({len(D)} tasks)")
    ev = {t: r["best_evaluation"] for t, r in D.items()}
    one = [r["candidates"][0] for r in D.values()]
    r0 = [max(c["score"] for c in round0(r)) for r in D.values()]
    best = [e["overall"] for e in ev.values()]
    print("gate passed  one-shot / round-0 / loop:",
          sum(c["gate_passed"] for c in one),
          sum(any(c["gate_passed"] for c in round0(r)) for r in D.values()),
          sum(e["gate_passed"] for e in ev.values()))
    print("score median/mean  one-shot", ms([c["score"] for c in one]),
          "| round-0", ms(r0), "| loop", ms(best))
    gain = [b - a for a, b in zip(r0, best)]
    spread = [max(c["score"] for c in round0(r)) - min(c["score"] for c in round0(r)) for r in D.values()]
    print(f"loop gain over round 0: mean {st.mean(gain):+.3f}, improved {sum(g > 1e-6 for g in gain)}; "
          f"round-0 seed range median {st.median(spread):.3f}")
    for b in ("keep", "follow", "world", "weighted_sum_baseline"):
        print(f"  {b:22s}", ms([e["branch_scores"].get(b) for e in ev.values()]))
    crit = collections.defaultdict(list)
    for e in ev.values():
        for k, v in e["critics"].items():
            if v["score"] is not None:
                crit[k].append(v["score"])
    print("  critics (median):", " ".join(f"{k}={st.median(v):.2f}" for k, v in crit.items()))
    print("  issues at best:", dict(collections.Counter(i for e in ev.values() for i in e["issues"])))
    acts = collections.defaultdict(list)
    for r in D.values():
        for rd in r["rounds"][1:]:
            acts[rd["action"]].append((rd["accepted"], rd["round_best"] - rd["best_before"]))
    for a, v in sorted(acts.items(), key=lambda kv: -len(kv[1])):
        print(f"  action {a:18s} n={len(v):3d} accepted={sum(x for x, _ in v):3d} "
              f"mean delta={st.mean(d for _, d in v):+.3f}")
    for key in ("target", "subject"):
        g = collections.defaultdict(list)
        for r in D.values():
            s = r["spec"]
            k = s["target_bg"]["key"] if key == "target" else s["subject"] + ("_submerged" if s["submerged"] else "")
            g[k].append(r["best_evaluation"]["overall"])
        print(f"  by {key}:", " ".join(f"{k}={st.median(v):.2f}(n={len(v)},gated={sum(x == 0 for x in v)})"
                                     for k, v in sorted(g.items())))
    cands = [c for r in D.values() for c in r["candidates"]]
    hack = [c for c in cands if not c["gate_passed"] and c["branch_scores"].get("weighted_sum_baseline", 0) > 0.65]
    print(f"  candidates {len(cands)}, gate=0 {sum(not c['gate_passed'] for c in cands)}, "
          f"gate=0 but weighted sum > 0.65: {len(hack)}")


def pairwise(R: dict[str, dict]) -> None:
    names = list(R)
    common = sorted(set.intersection(*(set(D) for D in R.values())))
    print(f"\n===== pairwise on {len(common)} common tasks")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = {t: R[a][t]["best_evaluation"] for t in common}
            sb = {t: R[b][t]["best_evaluation"] for t in common}
            line = f"{a} > {b}: {sum(sa[t]['overall'] > sb[t]['overall'] for t in common)}"
            for br in ("keep", "follow", "world"):
                line += f" | {br} {sum(sa[t]['branch_scores'][br] > sb[t]['branch_scores'][br] for t in common)}"
            print(line)
            for k in sa[common[0]]["critics"]:
                xa = [sa[t]["critics"][k]["score"] for t in common]
                xb = [sb[t]["critics"][k]["score"] for t in common]
                if None in xa or None in xb:
                    continue
                print(f"    {k:16s} {a} wins {sum(x > y + 1e-6 for x, y in zip(xa, xb)):3d}  "
                      f"{b} wins {sum(y > x + 1e-6 for x, y in zip(xa, xb)):3d}")


def write_summary(runs_dir: Path, name: str, D: dict[str, dict]) -> None:
    cols = ["task_id", "sample", "target", "submerged", "editor", "best_score", "gate_passed", "keep", "follow",
            "world", "weighted_sum_baseline", "first_score", "edits_used", "rounds", "stopped", "issues"]
    with open(runs_dir / name / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t, r in D.items():
            e, s, b = r["best_evaluation"], r["spec"], r["best_evaluation"]["branch_scores"]
            w.writerow([t, s["sample_id"], s["target_bg"]["key"], int(s["submerged"]), r["editor"],
                        round(e["overall"], 4), e["gate_passed"], round(b["keep"], 4), round(b["follow"], 4),
                        round(b["world"], 4), round(b["weighted_sum_baseline"], 4),
                        round(max(c["score"] for c in round0(r)), 4), r["edits_used"], len(r["rounds"]),
                        r["stopped_reason"], " ".join(e["issues"])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--csv", action="store_true", help="rebuild runs/<name>/summary.csv from report.json")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    R = {n: load(runs_dir, n) for n in args.names}
    for n, D in R.items():
        summarise(n, D)
        if args.csv:
            write_summary(runs_dir, n, D)
    if len(R) > 1:
        pairwise(R)


if __name__ == "__main__":
    main()
