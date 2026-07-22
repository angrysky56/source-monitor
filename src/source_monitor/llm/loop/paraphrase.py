"""F24 — "murmuration of a claim": faithful paraphrase families for the detector.

Motivation (the dual-representation idea, from the F22/F23 arc). A hallucinated
claim is a fragile fixed point: it reads as fine locally, but it is not robustly
*supported*. So its surprisal should be UNSTABLE when you reword the support,
whereas a genuinely grounded claim should be STABLE. The murmuration analogy:
a signal invisible in any single instance (one forward pass) becomes visible when
you look at the DISTRIBUTION over a family of related instances (paraphrases).

Where F23 perturbed the model's WEIGHTS and re-scored a fixed span, F24 perturbs
the PHRASING of the context and re-scores a fixed span. Two invariants make the
dispersion meaningful rather than noise, and both are enforced/tested:

  1. FACT-FAITHFUL. entity_prose is procedural: facts (object, per-turn location,
     moves, removal, the planted lie) are separable from surface wording. A family
     member re-renders each support turn from a different phrasing template while
     holding its (object, location_text) fixed — same facts, different words.
  2. SPAN-INVARIANT. When scoring assistant span s, that span's tokens are held
     BYTE-IDENTICAL across its family; only the *other* support turns are reworded.
     So the k scores for s differ only because its support was rephrased — which
     is exactly the quantity ("stability of s under support paraphrase") we want.

CAUTION: the phrasing pools are finite, so for large k members repeat and
dispersion is UNDER-stated (conservative). This is a detector-measurement tool,
not a generation path; it never changes what the model emits.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import torch

from source_monitor.llm.loop.monitor import build_context, span_scores
from source_monitor.llm.ood.base import OODTrace
from source_monitor.llm.task_render import Turn

# Phrasing pool — a superset of entity_prose's, so k up to ~6 gives distinct
# rewordings. Keeps the (object, location) slots; only the frame varies.
#
# We reword ONLY the assistant's acknowledgements. User instruction turns are held
# byte-identical on purpose: their true location is the GROUND TRUTH a planted lie
# contradicts. An earlier version re-derived the instruction's location from the
# following ack's location_text — but for a corrupted ack that is the WRONG
# location, so it silently rewrote the instruction to agree with the lie and
# erased the very inconsistency the detector measures (F24 debug: lie fam_mean
# collapsed 25.4 -> 0.0). Rewording acks alone keeps facts intact.
ACK_POOL = (
    "Done — the {o} is in {l}.",
    "Okay, the {o} is now in {l}.",
    "Got it; the {o} sits in {l}.",
    "Sure, the {o} is in {l} now.",
    "The {o} is resting in {l}.",
    "Noted — the {o} is in {l}.",
    "Right, the {o} went to {l}.",
)

# Arithmetic running-total frames — the number slot {n} is held; only the frame
# varies. Same faithfulness rule as ACK_POOL: reword the assistant's own report,
# never the user's ground-truth operation.
RT_POOL = (
    "Running total: {n}.",
    "The total is now {n}.",
    "That brings us to {n}.",
    "Now at {n}.",
    "So far: {n}.",
    "The running total is {n}.",
    "That makes {n}.",
)

_QUERY_PREFIX = "Where is the "
_QUERY_SUFFIX = " now?"


def _query_object(trace: OODTrace) -> str | None:
    """Recover the object string from the fixed-template query turn.

    The query is always ``"Where is the {obj} now?"`` (entity_prose), so this is
    an exact parse, not a heuristic. Returns None if no query turn is found (the
    trace is then left un-reworded, a safe no-op).
    """
    for t in trace.turns:
        if t.role == "user" and t.content.startswith(_QUERY_PREFIX) and t.content.endswith(
            _QUERY_SUFFIX
        ):
            return t.content[len(_QUERY_PREFIX) : -len(_QUERY_SUFFIX)]
    return None


def _is_removal(turn: Turn) -> bool:
    """A removal turn carries no location and has only one phrasing — hold it."""
    c = turn.content.lower()
    return "away" in c or "has been removed" in c or "removed" in c


def rerender_entity_prose(
    trace: OODTrace, hold_turn_index: int, rng: random.Random
) -> OODTrace:
    """entity_prose family: reword the assistant's acks, hold everything else.

    Held byte-identical: system, every user turn (the ground-truth instructions),
    the final claim, removal turns, and ``hold_turn_index`` (the scored span).
    Every OTHER assistant acknowledgement is re-rendered from a random ACK_POOL
    frame with its ORIGINAL object and location_text — so facts are preserved
    exactly (a planted lie keeps its wrong location) and only wording moves.

    Args:
        trace: The source trace.
        hold_turn_index: Turn to keep unchanged (the scored span).
        rng: Seeded RNG for reproducible template choices.

    Returns:
        A new OODTrace; the original is untouched.
    """
    obj = _query_object(trace)
    claim_ti = trace.claim.turn_index
    if obj is None:
        return trace  # cannot parse → safe no-op

    new: list[Turn] = []
    for i, t in enumerate(trace.turns):
        reword = (
            t.role == "assistant"
            and i != hold_turn_index
            and i != claim_ti
            and t.location_text is not None
            and not _is_removal(t)
        )
        if reword:
            # Re-render the ack; PRESERVE is_corrupted + location_text, so a planted
            # lie stays a confident FALSE statement, just reworded.
            new.append(
                replace(t, content=rng.choice(ACK_POOL).format(o=obj, l=t.location_text))
            )
        else:
            new.append(replace(t))

    return replace(trace, turns=new)


def rerender_arithmetic(
    trace: OODTrace, hold_turn_index: int, rng: random.Random
) -> OODTrace:
    """arithmetic family: reword the running-total reports, hold the numbers.

    Held byte-identical: system, every user op turn (the ground-truth operations),
    the final claim, and ``hold_turn_index``. Every OTHER "Running total: n"
    emission is re-rendered from an RT_POOL frame with its ORIGINAL number
    (location_text), so a corrupted mid total keeps its wrong value and only the
    surrounding phrasing changes. Long-ξ: rewording the support genuinely changes
    what the model must integrate to re-derive the total.
    """
    claim_ti = trace.claim.turn_index
    new: list[Turn] = []
    for i, t in enumerate(trace.turns):
        reword = (
            t.role == "assistant"
            and i != hold_turn_index
            and i != claim_ti
            and t.location_text is not None
        )
        if reword:
            new.append(replace(t, content=rng.choice(RT_POOL).format(n=t.location_text)))
        else:
            new.append(replace(t))
    return replace(trace, turns=new)


# entity_prose is the default; ``rerender`` kept as an alias for back-compat.
rerender = rerender_entity_prose
REWORDERS = {
    "entity_prose": rerender_entity_prose,
    "arithmetic": rerender_arithmetic,
}


@torch.no_grad()
def family_span_stats(
    model: Any,
    tokenizer: Any,
    trace: OODTrace,
    device: str,
    k: int = 6,
    base_seed: int = 0,
    identity: bool = False,
    rerender_fn: Any = rerender_entity_prose,
) -> tuple[list[float], list[float], list[tuple[int, int, int]]]:
    """Per assistant span: (mean, std) of slot surprisal over its paraphrase family.

    For each context assistant emission ``a`` (in ``build_context`` order), builds
    ``k`` family members that hold span ``a`` byte-identical and reword the rest,
    scores span ``a`` in each, and returns the across-member mean and std.

    ``identity=True`` builds the family from k EXACT copies (no rewording): the
    control. It must return std == 0 everywhere and mean == the single-pass score,
    which is the F24 harness-integrity check (mirrors F23's sigma=0 control).

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        trace: The trace to score.
        device: Torch device string.
        k: Family size (paraphrases per span).
        base_seed: Seed base; member/span get distinct streams.
        identity: If True, disable rewording (control).

    Returns:
        ``(means, stds, asst)`` where means/stds are per assistant span and
        ``asst`` is ``build_context``'s ``[(turn_index, start, end)]`` list, so the
        caller can map ``corrupt_turn_index`` to a span ordinal.
    """
    _ids, _spans, asst = build_context(tokenizer, trace, device)
    m = len(asst)
    means: list[float] = []
    stds: list[float] = []

    for a in range(m):
        hold_ti = asst[a][0]
        vals: list[float] = []
        for j in range(k):
            if identity:
                member = trace
            else:
                rng = random.Random(base_seed * 7919 + a * 131 + j)
                member = rerender_fn(trace, hold_ti, rng)
            ids2, spans2, _asst2 = build_context(tokenizer, member, device)
            scores = span_scores(model, ids2, spans2)
            vals.append(scores[a])
        mu = sum(vals) / k
        var = sum((v - mu) ** 2 for v in vals) / k
        means.append(mu)
        stds.append(var**0.5)

    return means, stds, asst
