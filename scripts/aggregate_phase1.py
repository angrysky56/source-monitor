"""Aggregate Phase 1 (OOD transfer) results and evaluate the pre-registered gate.

Reads results/llm_phase1_results.jsonl; reports mean +/- std over seeds of
matched-surface (value / negation) and pooled AUROC per domain x model x scorer,
for the mean and value aggregations, plus calibration ECE. Then checks
P-1.1 / P-1.2 / P-1.3.

Usage: python scripts/aggregate_phase1.py [path]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/llm_phase1_results.jsonl"
rows = [json.loads(x) for x in open(PATH)]

# (domain, model, scorer, agg) -> metric -> [values over seeds]
acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in rows:
    for scorer, aggs in r["scorers"].items():
        for agg, m in aggs.items():
            key = (r["domain"], r["model_name"][-4:], scorer, agg)
            for metric in ("matched_value_auroc", "matched_negation_auroc",
                           "pooled_auroc", "calib_ece"):
                v = m.get(metric)
                if v is not None and v == v:  # drop NaN
                    acc[key][metric].append(v)


def ms(vals):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "  --  "


domains = sorted({k[0] for k in acc})
models = sorted({k[1] for k in acc})

for agg in ("mean", "value"):
    print(f"\n{'='*90}\n{agg.upper()} aggregation | matched value / matched negation / pooled AUROC\n{'='*90}")
    print(f"{'domain':<14}{'model':<6}{'scorer':<12}{'value':<16}{'negation':<16}{'pooled':<16}ECE")
    for d in domains:
        for mdl in models:
            for sc in ("raw", "contrastive"):
                k = (d, mdl, sc, agg)
                if k not in acc:
                    continue
                a = acc[k]
                print(f"{d:<14}{mdl:<6}{sc:<12}"
                      f"{ms(a['matched_value_auroc']):<16}{ms(a['matched_negation_auroc']):<16}"
                      f"{ms(a['pooled_auroc']):<16}{ms(a['calib_ece'])}")

print(f"\n{'='*90}\nGATE\n{'='*90}")
best_model = "4B" if any(k[1] == "4B" for k in acc) else max(models)
print(f"(evaluating on largest model present: {best_model})")


def best_value(domain):
    """Best matched-value AUROC (mean agg) across scorers at best_model."""
    out = []
    for sc in ("raw", "contrastive"):
        k = (domain, best_model, sc, "mean")
        if k in acc and acc[k]["matched_value_auroc"]:
            out.append(np.mean(acc[k]["matched_value_auroc"]))
    return max(out) if out else float("nan")


print("\nP-1.1  positive-claim transfer holds (matched value AUROC >= .85):")
for d in ("entity_prose", "arithmetic", "factual_qa"):
    v = best_value(d)
    print(f"  {d:<14}: {v:.3f}  [{'PASS' if v >= 0.85 else 'FAIL'}]")

print("\nP-1.2  contrastive+mean beats raw on negation by >= .05 (arithmetic, factual_qa):")
for d in ("arithmetic", "factual_qa"):
    kr = (d, best_model, "raw", "mean")
    kc = (d, best_model, "contrastive", "mean")
    if kr in acc and kc in acc:
        r = np.mean(acc[kr]["matched_negation_auroc"])
        c = np.mean(acc[kc]["matched_negation_auroc"])
        print(f"  {d:<14}: raw {r:.3f} -> contrastive {c:.3f}  (Δ{c-r:+.3f})  "
              f"[{'PASS' if c-r >= 0.05 else 'no'}]")

print("\nP-1.3  free-text floor (code_trace raw matched value AUROC >= .70):")
k = ("code_trace", best_model, "raw", "mean")
if k in acc and acc[k]["matched_value_auroc"]:
    v = np.mean(acc[k]["matched_value_auroc"])
    print(f"  code_trace    : {v:.3f}  [{'PASS' if v >= 0.70 else 'FAIL'}]")
