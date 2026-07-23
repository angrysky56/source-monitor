"""F26 — the factual leg: answer-consistency under query paraphrase.

F25 located the gap precisely. Single-pass surprisal CEILINGS when the lie
contradicts context, and (F19) COLLAPSES to ~chance on pure RECALL, where the
claim has no in-context support. Recall is where hallucinations live, and it needs
a different signal. This module implements the consistency signal — the
SelfCheckGPT intuition, adapted to seer's teacher-forced loop:

    Paraphrase the QUESTION k ways; for each, read which candidate answer the
    model prefers (lowest surprisal). A fact the model KNOWS yields the same
    preferred answer whatever the framing (stable); a confabulation flips
    (unstable). STABILITY, not surprisal, is the factual-leg detector.

No free generation: the preferred answer is read off teacher-forced candidate
scoring (base.raw_claim_score over base.make_variant), so it composes with the
rest of the loop and stays cheap and deterministic. This is the second leg of a
two-leg routed monitor — context-derivable claims -> surprisal (near-perfect,
F21e/F25), factual claims -> this.

v1 caveat: QFRAMES are shallow wrappers (they change the tokens AROUND the
question, not its structure). A richer family — true structural paraphrase or
model-generated rewordings — is the obvious next step and would test the signal
harder.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any

import torch

from source_monitor.llm.ood.base import make_variant, raw_claim_score

# Meaning-preserving question frames; QFRAMES[0] is the identity (control).
QFRAMES = (
    "{q}",
    "Please answer: {q}",
    "Quick question: {q}",
    "Here's a question — {q}",
    "I'd like to know: {q}",
    "Can you tell me, {q}",
    "Answer concisely: {q}",
)


def _question_turn_index(trace: Any) -> int:
    """Index of the last user turn before the claim — the question being answered."""
    ci = trace.claim.turn_index
    for i in range(ci - 1, -1, -1):
        if trace.turns[i].role == "user":
            return i
    raise ValueError("no question turn found before the claim")


def paraphrase_query(trace: Any, frame: str) -> Any:
    """Return a copy whose question turn is wrapped by ``frame``; else identical.

    Only the question's surrounding words change — the question itself (and every
    other turn, including the candidate machinery) is held — so a change in the
    model's preferred answer is attributable to framing alone.
    """
    qi = _question_turn_index(trace)
    base_q = trace.turns[qi].content
    new_turns = list(trace.turns)
    new_turns[qi] = replace(trace.turns[qi], content=frame.format(q=base_q))
    return replace(trace, turns=new_turns)


def _frames(k: int) -> list[str]:
    """First k frames (deterministic; identity always included as the control)."""
    if k <= len(QFRAMES):
        return list(QFRAMES[:k])
    reps = (k + len(QFRAMES) - 1) // len(QFRAMES)
    return (list(QFRAMES) * reps)[:k]


@torch.no_grad()
def preferred_candidate(model: Any, tok: Any, trace: Any, device: str) -> int:
    """Index of the VALUE candidate the model finds most probable (value slot).

    Two corrections the F26 smoke forced:
    - Score the VALUE SLOT only (``value_only_neglogp``), not the whole claim —
      whole-claim mean is length-biased toward the long negation.
    - EXCLUDE negation/abstention candidates. The system prompt primes "no reliable
      record", so under teacher-forcing the hedge has uniformly low surprisal and
      is "preferred" for every question, masking whether the model actually
      retrieves the fact. Consistency of the preferred VALUE across paraphrases is
      the retrieval signal; abstention is a separate axis.
    """
    surfaces = trace.claim.candidate_surfaces or []
    idxs = [i for i, s in enumerate(surfaces) if s == "value"] or list(
        range(len(trace.claim.candidate_contents))
    )
    scores = {
        i: raw_claim_score(model, tok, make_variant(trace, i), device).value_only_neglogp
        for i in idxs
    }
    return min(idxs, key=lambda i: scores[i])


@torch.no_grad()
def answer_stability(
    model: Any, tok: Any, trace: Any, device: str, k: int = 6
) -> dict:
    """Consistency of the model's preferred answer across k question frames.

    Returns:
        dict with ``stability`` (modal fraction in [1/k, 1]; 1.0 = same answer
        every framing), ``entropy`` (nats of the preferred-answer distribution),
        ``modal_index``, ``modal_is_value`` (True if the stable pick is a value,
        not the negation/abstain), ``modal_correct``, and the raw ``prefs`` list.
    """
    frames = _frames(k)
    prefs = [
        preferred_candidate(model, tok, paraphrase_query(trace, f), device)
        for f in frames
    ]
    counts = Counter(prefs)
    modal, cnt = counts.most_common(1)[0]
    total = len(prefs)
    entropy = -sum((v / total) * math.log(v / total) for v in counts.values())
    surfaces = trace.claim.candidate_surfaces or []
    return {
        "stability": cnt / total,
        "entropy": entropy,
        "modal_index": modal,
        "modal_is_value": bool(surfaces and surfaces[modal] == "value"),
        "modal_correct": modal == trace.claim.correct_index,
        "prefs": prefs,
    }
