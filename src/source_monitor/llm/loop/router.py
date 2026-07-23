"""F27 — The Router: combining surprisal (Leg 1) and sampled consistency (Leg 2).

Dispatches each assistant claim to the appropriate detector leg based on whether
its truth is derivable from the prior context:
- Context-derivable claims -> Leg 1 (retrospective surprisal)
- Context-underivable / factual claims -> Leg 2 (sampled consistency)

CRITICAL INVARIANTS (F27 Hand-off Constraints):
1. CONTENT-ONLY CLASSIFICATION: is_context_derivable MUST NOT read trace.meta
   (e.g., "grounded"). That is an oracle label. Classification is based strictly
   on text overlap / entity presence in prior context turns.
2. INDEPENDENT LEG CALIBRATION: Surprisal (nats) and consistency (agreement /
   distinct_ratio) operate on different scales. Each leg is calibrated
   independently to produce a binary flag; the router selects which leg's
   binary flag to evaluate.
3. IDENTITY CONTROLS: On all-derivable data (entity_prose/arithmetic), 100% of
   claims route to Leg 1 and output is identical to Leg-1-only. On all-factual
   data (factual_qa), 100% of claims route to Leg 2 and output is identical to
   Leg-2-only.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from source_monitor.llm.loop.consistency import _normalize_answer, sampled_consistency
from source_monitor.llm.loop.monitor import build_context, flag_index, span_scores

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _extract_context_text(trace: Any) -> str:
    """Renders prior user/assistant dialogue turns (excluding system instructions) into a lowercased string."""
    ci = trace.claim.turn_index
    # Include non-system turns or turns prior to the question
    turns = [t for t in trace.turns[:ci] if t.role in ("user", "assistant")]
    return " ".join(t.content.lower() for t in turns)


def _tokenize_words(text: str) -> set[str]:
    """Extracts non-empty alnum word tokens."""
    return set(filter(None, _NON_ALNUM.split(text.lower())))


def is_context_derivable(trace: Any, span: Any = None) -> bool:
    """Decide if the claim's value/entity is present in or derivable from prior context.

    CRITICAL: MUST NOT READ trace.meta (e.g. 'grounded'). Content-only evaluation.

    Algorithm:
    1. Extract prior user/assistant dialogue context text (excluding system instructions).
    2. Extract VALUE candidate surfaces (excluding negation/abstention candidates).
    3. Use word-boundary matching to check if candidate values appear in dialogue context.
    4. Return True if a value candidate is present in context, else False.
    """
    ctx_text = _extract_context_text(trace)
    if not ctx_text.strip():
        return False

    ctx_words = _tokenize_words(ctx_text)

    claim = trace.claim
    values = getattr(claim, "candidate_values", None) or []
    surfaces = getattr(claim, "candidate_surfaces", None) or []

    # Exclude negation/abstention candidates ("no reliable record", etc.)
    valid_values: list[str] = []
    for i, val in enumerate(values):
        if not val or len(val.strip()) == 0:
            continue
        if surfaces and i < len(surfaces) and surfaces[i] == "negation":
            continue
        if "no reliable record" in val.lower():
            continue
        valid_values.append(val.strip().lower())

    if not valid_values:
        query_obj = getattr(trace, "query_object", "") or ""
        if query_obj and re.search(r"\b" + re.escape(query_obj.lower()) + r"\b", ctx_text):
            return True
        return False

    for val in valid_values:
        # Exact word boundary match in context (prevents "fe" matching "confidence")
        pattern = r"\b" + re.escape(val) + r"\b"
        if re.search(pattern, ctx_text):
            return True
        # Word token subset match for multi-word values (e.g., "the cellar" -> {"the", "cellar"})
        val_words = _tokenize_words(val)
        # Exclude common stop words from single requirement
        val_content_words = {w for w in val_words if w not in {"the", "a", "an", "in", "on", "at", "to", "is"}}
        if val_content_words and val_content_words.issubset(ctx_words):
            return True

    return False


def route(trace: Any, span: Any = None) -> str:
    """Returns 'surprisal' for context-derivable claims, 'consistency' for factual claims."""
    return "surprisal" if is_context_derivable(trace, span) else "consistency"


@torch.no_grad()
def evaluate_routed_trace(
    model: Any,
    tok: Any,
    trace: Any,
    device: str,
    surprisal_floor: float | None = None,
    surprisal_k: float = 1.5,
    consistency_k: int = 5,
    consistency_temp: float = 0.8,
    consistency_threshold: float = 0.5,
    seed: int = 42,
    apply_precision_weighting: bool = False,
) -> dict[str, Any]:
    """Evaluates a trace using the routed leg's binary flag.

    Returns a dict containing branch decisions, raw scores, binary flags, and audit info.
    """
    derivable = is_context_derivable(trace)
    branch = route(trace)

    # 1. Leg 1: Retrospective Surprisal
    input_ids, spans, asst = build_context(tokenizer=tok, trace=trace, device=device)
    scores = span_scores(model, input_ids, spans) if asst else []
    max_surprisal = max(scores) if scores else 0.0

    fi = flag_index(
        scores,
        k=surprisal_k,
        floor=surprisal_floor,
        mode="absolute" if surprisal_floor is not None else "zscore",
    )
    surprisal_flag = fi is not None

    # 2. Leg 2: Sampled Consistency
    cons_res = sampled_consistency(
        model, tok, trace, device, k=consistency_k, temperature=consistency_temp, seed=seed
    )
    distinct_ratio = cons_res["distinct_ratio"]
    agreement = cons_res["agreement"]
    answers = cons_res["answers"]

    # Factual claim verification: check if emitted/claimed value is in modal sampled consensus
    values = trace.claim.candidate_values or []
    claimed_val = values[trace.claim.correct_index].lower() if values else ""
    norm_answers = [_normalize_answer(a) for a in answers]

    # Consistency flag: fires if answer set has high distinct ratio OR claimed value missing from consensus
    consistency_flag = bool(distinct_ratio > consistency_threshold or (
        claimed_val and not any(claimed_val in na for na in norm_answers)
    ))

    # Precision-weighting ablation (optional multiplier for high-confidence x underivable)
    danger_multiplier = 1.0
    if apply_precision_weighting and not derivable:
        # High confidence = low surprisal
        confidence = max(0.0, 1.0 - max_surprisal / 10.0)
        danger_multiplier = 1.0 + confidence

    # Final routed decision selects the active leg's binary flag
    routed_flag = surprisal_flag if branch == "surprisal" else consistency_flag

    return {
        "derivable": derivable,
        "branch": branch,
        "raw_surprisal": max_surprisal,
        "surprisal_flag": surprisal_flag,
        "distinct_ratio": distinct_ratio,
        "agreement": agreement,
        "consistency_flag": consistency_flag,
        "danger_multiplier": danger_multiplier,
        "routed_flag": routed_flag,
        "sampled_answers": answers,
        "claim_value": claimed_val,
        "context_snippet": _extract_context_text(trace)[:80],
    }
