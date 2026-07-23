"""Throwaway calibration probe (Build B §4): does the confabulation zone populate?

Runs sampled answers over the obscure_facts seed bank and bins each question into
Known / Confabulation / Refusal / Borderline, so we can SEE the zone split and raw
samples before building the F28 eval. Not a permanent runner.
"""
from __future__ import annotations

import re
from collections import Counter

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.consistency import _is_hedge, _normalize_answer, sample_answers
from source_monitor.llm.ood import obscure_facts

cfg = Phase3Config()
K = 6
model, tok, _ = load_model(cfg.model_name, device=cfg.device, dtype=cfg.dtype, enable_thinking=False)
model.eval()

traces = obscure_facts.generate(42, 100)  # all items


def correct(ans: str, acceptable: list[str]) -> bool:
    a = ans.lower()
    return any(re.search(r"\b" + re.escape(x) + r"\b", a) for x in acceptable)


rows = []
for tr in traces:
    ans = sample_answers(model, tok, tr, cfg.device, k=K, temperature=0.8, seed=7)
    acc = tr.meta["acceptable"]
    cr = sum(correct(a, acc) for a in ans) / K
    hr = sum(_is_hedge(a) for a in ans) / K
    dr = len({_normalize_answer(a) for a in ans}) / K
    if hr >= 0.5:
        zone = "REFUSAL"
    elif cr >= 0.66 and dr <= 0.34:
        zone = "KNOWN"
    elif cr <= 0.34:
        zone = "CONFAB"
    else:
        zone = "BORDER"
    rows.append({"q": tr.turns[1].content, "cat": tr.meta["category"], "canon": tr.meta["canonical"],
                 "cr": cr, "hr": hr, "dr": dr, "zone": zone, "ans": ans})

print("\n==== ZONE SPLIT (n=%d, k=%d) ====" % (len(rows), K))
zc = Counter(r["zone"] for r in rows)
for z in ("KNOWN", "CONFAB", "REFUSAL", "BORDER"):
    print(f"  {z:<8} {zc.get(z,0):>3}")
print("  by category:")
for cat in sorted({r["cat"] for r in rows}):
    cc = Counter(r["zone"] for r in rows if r["cat"] == cat)
    print(f"    {cat:<15} " + " ".join(f"{z}:{cc.get(z,0)}" for z in ("KNOWN","CONFAB","REFUSAL","BORDER")))

for z in ("CONFAB", "KNOWN", "REFUSAL", "BORDER"):
    ex = [r for r in rows if r["zone"] == z][:3]
    if ex:
        print(f"\n---- {z} examples ----")
        for r in ex:
            print(f"  Q: {r['q']}  (truth={r['canon']})  cr={r['cr']:.2f} hr={r['hr']:.2f} dr={r['dr']:.2f}")
            print(f"     samples: {[a[:22] for a in r['ans']]}")
