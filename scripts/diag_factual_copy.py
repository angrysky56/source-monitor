"""Diagnostic: can retrospective surprisal detect a wrong answer that is stated
in-context? Isolates in-context COPYING from world-knowledge PRIOR interference.

Four conditions, each scored as value-token raw AUROC (corrupt vs genuine) over a
set of items:
  A nonsense-copy      : "The secret code is Zebra." then query it (no prior).
  B realfact-plain     : "The capital of France is Paris." then query (natural).
  C realfact-meta      : 'For reference: the answer to "..." is Paris.' (as generator).
  D realfact-nogrounding: query only (pure recall).

If A ~1.0, the model CAN copy an in-context fact into separable surprisal, so the
factual_qa/grounded failure is prior interference / phrasing, not a scoring bug.
"""

from __future__ import annotations

import random

from source_monitor.llm.cache import load_model
from source_monitor.llm.ood.base import OODClaim, OODTrace, raw_claim_score
from source_monitor.llm.task_render import Turn
from source_monitor.metrics import auroc

DEVICE, MODEL = "cuda", "Qwen/Qwen3-1.7B"
SYS = "You answer questions concisely based on the conversation."
NONSENSE = ["Zebra", "Quartz", "Marimba", "Kelp", "Vellum", "Onyx", "Dinghy",
            "Pomelo", "Gantry", "Bramble", "Cobalt", "Wisp"]
FACTS = [("What is the capital of France?", "Paris", "Lyon"),
         ("What is the capital of Japan?", "Tokyo", "Osaka"),
         ("What is the chemical symbol for gold?", "Au", "Ag"),
         ("Who wrote 'Hamlet'?", "Shakespeare", "Marlowe"),
         ("Which planet is closest to the Sun?", "Mercury", "Venus"),
         ("What is the capital of Egypt?", "Cairo", "Giza"),
         ("What is the capital of Canada?", "Ottawa", "Toronto"),
         ("What is the chemical symbol for iron?", "Fe", "Ir")]


def _trace(ctx_turns, query, answer):
    turns = [Turn(role="system", content=SYS, is_self=False, step_index=None)]
    turns += ctx_turns
    turns.append(Turn(role="user", content=query, is_self=False, step_index=9))
    turns.append(Turn(role="assistant", content=f"{answer}.", is_self=True, step_index=9,
                      claim_surface="value", location_text=answer))
    return OODTrace(domain="diag", turns=turns,
                    claim=OODClaim(turn_index=len(turns) - 1, free_surface="value"))


def score_condition(model, tok, items, build):
    gen, cor = [], []
    for (q, right, wrong) in items:
        ctx = build(q, right)
        gen.append(raw_claim_score(model, tok, _trace(ctx, q, right), DEVICE).value_only_neglogp)
        cor.append(raw_claim_score(model, tok, _trace(ctx, q, wrong), DEVICE).value_only_neglogp)
    return auroc(cor + gen, [1] * len(cor) + [0] * len(gen)), sum(gen) / len(gen), sum(cor) / len(cor)


def main():
    model, tok, _ = load_model(MODEL, device=DEVICE, dtype="bfloat16", enable_thinking=False)
    rng = random.Random(0)
    nonsense_items = [(f"What is the secret code?", w, rng.choice([x for x in NONSENSE if x != w]))
                      for w in NONSENSE]

    conds = {
        "A nonsense-copy": (nonsense_items,
                            lambda q, a: [Turn(role="user", content=f"The secret code is {a}.", is_self=False, step_index=0),
                                          Turn(role="assistant", content="Noted.", is_self=True, step_index=0)]),
        "B realfact-plain": (FACTS,
                             lambda q, a: [Turn(role="user", content=f"Note this fact: {a} is the answer to \"{q}\"", is_self=False, step_index=0),
                                           Turn(role="assistant", content="Noted.", is_self=True, step_index=0)]),
        "C realfact-meta": (FACTS,
                            lambda q, a: [Turn(role="user", content=f'For reference: the answer to "{q}" is {a}.', is_self=False, step_index=0),
                                          Turn(role="assistant", content="Noted.", is_self=True, step_index=0)]),
        "D realfact-none": (FACTS, lambda q, a: []),
    }
    print(f"{'condition':<20} {'value AUROC':<12} {'mean gen':<10} {'mean cor':<10}")
    for name, (items, build) in conds.items():
        a, g, c = score_condition(model, tok, items, build)
        print(f"{name:<20} {a:<12.3f} {g:<10.2f} {c:<10.2f}")
    print("DIAG OK")


if __name__ == "__main__":
    main()
