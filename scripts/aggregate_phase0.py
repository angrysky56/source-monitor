"""Aggregate Phase 0 / 0b results and evaluate the pre-registered gate.

Reads results/llm_phase0_results.jsonl, reports mean +/- std over seeds for each
(scoring, model, task), and evaluates the Phase 0b predictions P-0b.1/2/3 with the
raw->contrastive delta (the measured size of the absence-assertion prior bias).

Usage: python scripts/aggregate_phase0.py [path/to/results.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/llm_phase0_results.jsonl"

rows = [json.loads(line) for line in open(PATH)]

# --- completeness -----------------------------------------------------------
combos = defaultdict(list)
for r in rows:
    combos[(r["scoring"], r["model_name"], r["task_label"])].append(r["seed"])

print(f"Total rows: {len(rows)}")
print("Combos present (scoring | model | task -> seeds):")
for k in sorted(combos, key=str):
    print(f"  {k[0]:<11} | {k[1][-4:]:>5} | {k[2]:<7} -> {sorted(combos[k])}")
print()


def grp(scoring, model_suffix, task):
    return [
        r
        for r in rows
        if r["scoring"] == scoring
        and r["model_name"].endswith(model_suffix)
        and r["task_label"] == task
    ]


def stat(group, agg, field, corr):
    vals = [g[agg][field][corr] for g in group if corr in g[agg].get(field, {})]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(s):
    return f"{s[0]:.3f}±{s[1]:.3f}(n{s[2]})" if s else "  --  "


MODELS = [("0.6B", "0.6B"), ("1.7B", "1.7B")]
TASKS = ["primary", "hard"]

# --- full table (slot_only, the pre-registered aggregation) -----------------
print("=" * 92)
print("SLOT_ONLY aggregation  |  matched-surface AUROC (M) and pooled AUROC (P)")
print("=" * 92)
hdr = f"{'model/task':<14} {'scoring':<11} | {'ghost M / P':<22} {'misloc M / P':<22} {'phantom M / P':<22}"
print(hdr)
print("-" * len(hdr))
for _, msuf in MODELS:
    for task in TASKS:
        for scoring in ("raw", "contrastive"):
            g = grp(scoring, msuf, task)
            if not g:
                continue
            row = f"{msuf+'/'+task:<14} {scoring:<11} | "
            for corr in ("ghost", "mislocation", "phantom"):
                m = stat(g, "slot_only", "matched_surface_auroc", corr)
                p = stat(g, "slot_only", "pooled_auroc", corr)
                cell = f"{(m[0] if m else float('nan')):.3f}/{(p[0] if p else float('nan')):.3f}"
                row += f"{cell:<22} "
            print(row)
    print("-" * len(hdr))

# --- gate evaluation --------------------------------------------------------
print("\n" + "=" * 92)
print("PRE-REGISTERED GATE (Phase 0b)")
print("=" * 92)


def phantom_matched(scoring, msuf, task):
    return stat(grp(scoring, msuf, task), "slot_only", "matched_surface_auroc", "phantom")


print("\nP-0b.1  phantom matched-surface AUROC >= .90 under contrastive (1.7B, both configs)")
print("        [raw -> contrastive, slot_only; delta = measured prior-bias size]")
for task in TASKS:
    r = phantom_matched("raw", "1.7B", task)
    c = phantom_matched("contrastive", "1.7B", task)
    if r and c:
        d = c[0] - r[0]
        verdict = "PASS" if c[0] >= 0.90 else "FAIL"
        print(f"  1.7B/{task:<7}: raw {fmt(r)} -> contrastive {fmt(c)}  (delta {d:+.3f})  [{verdict}]")
# scaling context
print("        scaling (contrastive phantom matched, slot_only):")
for msuf in ("0.6B", "1.7B"):
    for task in TASKS:
        c = phantom_matched("contrastive", msuf, task)
        if c:
            print(f"          {msuf}/{task:<7}: {fmt(c)}")

print("\nP-0b.2  ghost pooled ~= ghost matched under contrastive (anomaly dissolves)")
for msuf in ("0.6B", "1.7B"):
    for task in TASKS:
        g = grp("contrastive", msuf, task)
        if not g:
            continue
        m = stat(g, "slot_only", "matched_surface_auroc", "ghost")
        p = stat(g, "slot_only", "pooled_auroc", "ghost")
        if m and p:
            gap = m[0] - p[0]
            print(f"  {msuf}/{task:<7}: matched {m[0]:.3f} vs pooled {p[0]:.3f}  (gap {gap:+.3f})")

print("\nP-0b.3  container types unchanged (>= .97 matched) under contrastive")
for msuf in ("0.6B", "1.7B"):
    for task in TASKS:
        g = grp("contrastive", msuf, task)
        if not g:
            continue
        gh = stat(g, "slot_only", "matched_surface_auroc", "ghost")
        ml = stat(g, "slot_only", "matched_surface_auroc", "mislocation")
        if gh and ml:
            ok = "PASS" if (gh[0] >= 0.97 and ml[0] >= 0.97) else "CHECK"
            print(f"  {msuf}/{task:<7}: ghost {gh[0]:.3f} | misloc {ml[0]:.3f}  [{ok}]")
