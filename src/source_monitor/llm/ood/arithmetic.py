"""Mid-OOD: multi-step arithmetic with a reported running total.

The assistant reports the running total after each op (context), then states the
final total (the corruptible claim). Ground truth is computed in Python. A
fraction of traces end in a non-integer division, where the correct claim is the
negation ("no whole-number result") — the phantom-analog for this domain.
Candidates are near-miss numbers (in-context-plausible) plus the negation.
"""

from __future__ import annotations

import random

from source_monitor.llm.ood.base import OODClaim, OODTrace
from source_monitor.llm.task_render import Turn

SYSTEM = (
    "You are a careful calculator. Track the running total as operations are "
    "applied and report it when asked. If a step has no whole-number result, "
    "say so."
)
NEG_CONTENT = "There is no whole-number result."
NEG_VALUE = "no whole-number result"


def _digit_swap(x: int) -> int:
    s = str(abs(x))
    if len(s) >= 2:
        s = s[1] + s[0] + s[2:]
        v = int(s)
        if v != abs(x):
            return v if x >= 0 else -v
    return x + 2


def _value_candidates(correct: int, rng: random.Random) -> tuple[list[int], int]:
    """Return (distinct near-miss values incl. correct, index of correct)."""
    cands = {correct, correct + 1, correct - 1, correct + 10, _digit_swap(correct)}
    cands.discard(None)  # safety
    ordered = sorted(cands)
    rng.shuffle(ordered)
    return ordered, ordered.index(correct)


def generate(seed: int, n: int, n_ops: int = 4,
             corrupt_mid: bool = False) -> list[OODTrace]:
    """Long-ξ running-total traces (F24d hard task).

    corrupt_mid plants a WRONG running total in a non-first mid emission (the true
    final total is unchanged, so the final claim stays correct — a mid-context lie
    the model must re-derive past, exactly as entity_prose does). Mid emissions
    always carry location_text = the number, so the detector scores the number
    slot consistently on planted and clean traces alike.
    """
    rng = random.Random(seed)
    traces: list[OODTrace] = []
    for _ in range(n):
        total = rng.randint(2, 20)
        turns = [Turn(role="system", content=SYSTEM, is_self=False, step_index=None)]
        turns.append(Turn(role="user", content=f"Start with {total}.",
                          is_self=False, step_index=0))
        turns.append(Turn(role="assistant", content=f"Running total: {total}.",
                          is_self=True, step_index=0, claim_surface="value",
                          location_text=str(total)))
        for k in range(1, n_ops):
            op = rng.choice(["+", "-", "*"])
            v = rng.randint(2, 9)
            if op == "+":
                total += v; u = f"Add {v}."
            elif op == "-":
                total -= v; u = f"Subtract {v}."
            else:
                total *= v; u = f"Multiply by {v}."
            turns.append(Turn(role="user", content=u, is_self=False, step_index=k))
            turns.append(Turn(role="assistant", content=f"Running total: {total}.",
                              is_self=True, step_index=k, claim_surface="value",
                              location_text=str(total)))

        # Plant a mid-context lie: corrupt a non-first running total. The chain is
        # still reported truthfully afterwards, so the wrong number contradicts its
        # neighbours and can only be caught by integrating the running total —
        # long-ξ, where single-pass surprisal is weakest (F20d) and support
        # paraphrase should bite hardest.
        corrupt_turn_index = None
        if corrupt_mid:
            mid = [i for i, t in enumerate(turns)
                   if t.role == "assistant" and t.location_text is not None
                   and t.step_index not in (0, None)]
            if mid:
                ti = rng.choice(mid)
                true_n = int(turns[ti].location_text)
                # Subtle near-miss (off-by-small): locally plausible, wrong only in
                # the running total. A big digit-swap is trivially caught by
                # single-pass and leaves no headroom to measure whether dispersion
                # adds anything (F24d). Off-by-±(1..3) forces integration to catch.
                wrong = true_n + rng.choice([-3, -2, -1, 1, 2, 3])
                turns[ti].content = f"Running total: {wrong}."
                turns[ti].location_text = str(wrong)
                turns[ti].is_corrupted = True
                corrupt_turn_index = ti

        negation_correct = rng.random() < 0.3
        if negation_correct:
            # Rig a non-integer division: correct answer is the negation.
            d = rng.choice([dd for dd in range(2, 8) if total % dd != 0] or [3])
            final_user = f"Now divide by {d}. What is the result?"
            # candidates: plausible rounded quotients (wrong) + negation (correct)
            wrongs = sorted({total // d, total // d + 1})
            contents = [f"The total is {w}." for w in wrongs] + [NEG_CONTENT]
            values = [str(w) for w in wrongs] + [NEG_VALUE]
            surfaces = ["value"] * len(wrongs) + ["negation"]
            correct = len(wrongs)
        else:
            final_user = "What is the total now?"
            vals, ci = _value_candidates(total, rng)
            contents = [f"The total is {v}." for v in vals] + [NEG_CONTENT]
            values = [str(v) for v in vals] + [NEG_VALUE]
            surfaces = ["value"] * len(vals) + ["negation"]
            correct = ci

        turns.append(Turn(role="user", content=final_user, is_self=False, step_index=n_ops))
        turns.append(Turn(role="assistant", content=contents[correct], is_self=True,
                          step_index=n_ops, is_corrupted=False,
                          claim_surface=surfaces[correct], location_text=values[correct]))
        claim = OODClaim(
            turn_index=len(turns) - 1,
            correct_index=correct,
            emitted_index=correct,
            candidate_contents=contents,
            candidate_values=values,
            candidate_surfaces=surfaces,
        )
        traces.append(OODTrace(domain="arithmetic", turns=turns, claim=claim,
                               meta={"negation_correct": negation_correct,
                                     "corrupt_turn_index": corrupt_turn_index}))
    return traces
