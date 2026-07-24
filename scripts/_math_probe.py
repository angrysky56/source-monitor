"""Throwaway probe: does advanced math populate the confabulation zone, and are the
wrong answers UNSTABLE (Leg 2 catches) or STABLE (the bromine blind spot)?

Python ground truth (no LLM-authoring risk), numeric answers (no surface-variation
mess). Thinking is off, so the model can't chain-of-thought → it blurts.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.consistency import _is_hedge, _normalize_answer, sample_answers
from source_monitor.llm.ood.base import OODClaim, OODTrace
from source_monitor.llm.task_render import Turn

SYSTEM = "You are a math assistant. Answer each question with just the final number."


def build_questions() -> list[tuple[str, str, int]]:
    rng = random.Random(1)
    q: list[tuple[str, str, int]] = []
    for _ in range(8):
        a, b = rng.randint(23, 97), rng.randint(23, 97)
        q.append(("mult", f"What is {a} times {b}?", a * b))
    for base, exp in [(2, 17), (3, 11), (7, 6), (5, 8), (2, 20), (6, 7)]:
        q.append(("power", f"What is {base} to the power of {exp}?", base**exp))
    for nn in [8, 9, 10, 11, 12]:
        q.append(("factorial", f"What is {nn} factorial?", math.factorial(nn)))
    for a, b, m in [(7, 5, 13), (4, 9, 17), (3, 8, 11), (5, 7, 19)]:
        q.append(("modexp", f"What is {a} to the power of {b}, modulo {m}?", pow(a, b, m)))
    for _ in range(5):
        a, b = rng.randint(48, 300), rng.randint(48, 300)
        q.append(("gcd", f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b)))
    for n, k in [(12, 5), (15, 4), (10, 3), (14, 6)]:
        q.append(("choose", f"What is {n} choose {k}?", math.comb(n, k)))
    return q


def make_trace(question: str, answer: int) -> OODTrace:
    turns = [
        Turn(role="system", content=SYSTEM, is_self=False, step_index=None),
        Turn(role="user", content=question, is_self=False, step_index=0),
    ]
    claim = OODClaim(turn_index=len(turns), correct_index=0, emitted_index=0,
                     candidate_contents=[str(answer)], candidate_values=[str(answer)],
                     candidate_surfaces=["value"])
    return OODTrace(domain="math_probe", turns=turns, claim=claim,
                    meta={"answer": str(answer)})


cfg = Phase3Config()
K = 6
model, tok, _ = load_model(cfg.model_name, device=cfg.device, dtype=cfg.dtype, enable_thinking=False)
model.eval()

rows = []
for cat, q, ans in build_questions():
    tr = make_trace(q, ans)
    samp = sample_answers(model, tok, tr, cfg.device, k=K, temperature=0.8, seed=7)
    cr = sum(bool(re.search(r"\b" + re.escape(str(ans)) + r"\b", s)) for s in samp) / K
    hr = sum(_is_hedge(s) for s in samp) / K
    dr = len({_normalize_answer(s) for s in samp}) / K
    if hr >= 0.5:
        zone = "REFUSAL"
    elif cr >= 0.66 and dr <= 0.34:
        zone = "KNOWN"
    elif cr <= 0.34 and dr <= 0.34:
        zone = "CONFAB_STABLE"   # wrong AND consistent -> Leg 2 BLIND
    elif cr <= 0.34:
        zone = "CONFAB_UNSTABLE"  # wrong AND varies -> Leg 2 catches
    else:
        zone = "BORDER"
    rows.append({"cat": cat, "q": q, "ans": ans, "cr": cr, "hr": hr, "dr": dr, "zone": zone, "samp": samp})

print("\n==== MATH ZONE SPLIT (n=%d, k=%d) ====" % (len(rows), K))
zc = Counter(r["zone"] for r in rows)
for z in ("KNOWN", "CONFAB_UNSTABLE", "CONFAB_STABLE", "REFUSAL", "BORDER"):
    print(f"  {z:<16} {zc.get(z,0):>3}")
print("  --> Leg-2 catchable = CONFAB_UNSTABLE; Leg-2 BLIND = CONFAB_STABLE")
for z in ("CONFAB_UNSTABLE", "CONFAB_STABLE", "BORDER", "KNOWN"):
    ex = [r for r in rows if r["zone"] == z][:3]
    if ex:
        print(f"\n---- {z} ----")
        for r in ex:
            print(f"  Q: {r['q']}  truth={r['ans']}  cr={r['cr']:.2f} dr={r['dr']:.2f}")
            print(f"     samples: {[s[:16] for s in r['samp']]}")
