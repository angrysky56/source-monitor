"""Aggregate Phase 2 results and evaluate the gate.

Reads results/llm_phase2_results.jsonl; reports competence / planted_acc / bsi /
detect_auroc per arm (mean +- std over seeds) and checks P-2.1 / P-2.3.
(P-2.2 held-out transfer is reported once the corrupt arm trains on one type and
is evaluated on another — see implementation_plan_phase2.md.)

Usage: python scripts/aggregate_phase2.py [path]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/llm_phase2_results.jsonl"
rows = [json.loads(x) for x in open(PATH)]

by_arm = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("competence", "planted_acc", "bsi", "detect_auroc"):
        v = r.get(k)
        if v is not None and v == v:
            by_arm[r["arm"]][k].append(v)


def ms(a):
    return f"{np.mean(a):.3f}±{np.std(a):.3f}" if a else "  --  "


print(f"{'arm':<12}{'competence':<16}{'planted_acc':<16}{'bsi':<16}{'detect_auroc'}")
for arm in ("base_noft", "base", "drop", "corrupt"):
    if arm not in by_arm:
        continue
    a = by_arm[arm]
    print(f"{arm:<12}{ms(a['competence']):<16}{ms(a['planted_acc']):<16}{ms(a['bsi']):<16}{ms(a['detect_auroc'])}")

print("\nGATE")
if "drop" in by_arm and "base" in by_arm:
    base_bsi = np.mean(by_arm["base"]["bsi"])
    drop_bsi = np.mean(by_arm["drop"]["bsi"])
    base_comp = np.mean(by_arm["base"]["competence"])
    drop_comp = np.mean(by_arm["drop"]["competence"])
    tax = base_comp - drop_comp
    p21 = drop_bsi <= 0.5 * base_bsi and tax <= 0.02
    print(f"P-2.1 hole-rehearsal halves bsi w/o competence tax: "
          f"base bsi {base_bsi:.3f} -> drop bsi {drop_bsi:.3f}; tax {tax:+.3f}  "
          f"[{'PASS' if p21 else 'FAIL'}]")
if "drop" in by_arm and "base_noft" in by_arm:
    base_det = np.mean(by_arm["base_noft"]["detect_auroc"])
    drop_det = np.mean(by_arm["drop"]["detect_auroc"])
    p23 = drop_det >= 0.95 * base_det
    print(f"P-2.3 detection survives LoRA: base {base_det:.3f} -> drop {drop_det:.3f}  "
          f"[{'PASS' if p23 else 'FAIL'}]")
