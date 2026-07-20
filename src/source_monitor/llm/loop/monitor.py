"""The closed source-monitor loop: detect -> flag -> excise -> regenerate -> grade.

Reuses the validated pieces unchanged: retrospective surprisal for detection
(Phases 0/1) and a holed attention mask for excision (the F20e operation). The
only new logic is the flagging rule, the mask construction, and free-text grading
— all pure functions, unit-tested on CPU.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from source_monitor.llm.provenance import tokenize_with_provenance
from source_monitor.llm.task_render import Trace
from source_monitor.llm.telemetry import (
    _ASSISTANT_HEADER,
    render_chatml,
    retrospective_surprisal,
)

# Absence is expressed with arbitrary adverbs ("is not CURRENTLY anywhere"), so a
# literal cue list silently mis-scores correct negations as abstains. Require the
# unambiguous absence word (nowhere / ...anywhere) with up to two words of slack.
# "removed" is deliberately NOT a cue: it co-occurs with locations ("was removed
# from the attic, now in the cellar") and would fire falsely.
_NEG_RE = re.compile(
    r"nowhere|(?:isn'?t|is not|not|no longer)\s+(?:\w+\s+){0,2}anywhere",
    re.IGNORECASE,
)

# Qwen3 emits <think>...</think> scaffolding by default when given a bare
# assistant header. Prefilling an EMPTY think block is the documented way to
# disable it. Without this the model spends the whole token budget reasoning and
# never emits an answer (the Phase 3 all-abstain bug). Note: cache.load_model's
# enable_thinking flag is metadata only — it never affected generation, which
# went unnoticed because Phases 0-2 were entirely teacher-forced.
THINK_OFF = "<think>\n\n</think>\n\n"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop think blocks so grading sees the ANSWER, not the reasoning.

    Critical for correctness: the reasoning trace names locations ("moved it to
    the cellar"), so an ungated grader would score the model's deliberation
    instead of its final claim.
    """
    text = _THINK_RE.sub(" ", text)
    if "<think>" in text:  # unterminated: budget ran out mid-reasoning
        text = text.split("<think>")[0]
    return text.strip()


def build_context(tokenizer: Any, trace, device: str):
    """Context = turns up to (not incl.) the final claim turn + generation prompt.

    Returns (input_ids [1, L], span annotations, asst_spans) where asst_spans is
    [(turn_index, start_token, end_token)] per assistant emission. Token indices
    come from the prefix tokenization and stay valid once the generation prompt is
    appended (same string prefix ⇒ same token prefix).
    """
    ctx_turns = trace.turns[: trace.claim.turn_index]
    pre = Trace(
        turns=ctx_turns, query_object="", ground_truth_final="", op_kinds=[], task=None
    )  # type: ignore[arg-type]
    _ids, spans = tokenize_with_provenance(tokenizer, pre, device)
    asst = [
        (i, s.start_token, s.end_token)
        for i, s in enumerate(spans)
        if s.kind == "assistant" and s.end_token > s.start_token
    ]
    text = render_chatml(ctx_turns) + _ASSISTANT_HEADER + THINK_OFF
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(device), spans, asst


@torch.no_grad()
def span_scores(model: Any, input_ids, spans) -> list[float]:
    """Value-only retrospective surprisal per assistant emission (the detector)."""
    return [
        s.slot_only_neglogp for s in retrospective_surprisal(model, input_ids, spans)
    ]


def flag_index(
    scores: list[float],
    k: float = 1.5,
    floor: float | None = None,
    mode: str = "zscore",
) -> int | None:
    """Flag the highest-scoring self-span, or None.

    mode="zscore"   : within-trace outlier (RELATIVE). Fires on almost every trace
                      when there are only a few spans — kept for comparison.
    mode="absolute" : score must exceed a floor calibrated on clean traces. This
                      is what makes the monitor quiet when nothing is wrong.
    mode="both"     : AND of the two.
    """
    n = len(scores)
    if n == 0:
        return None
    best = max(range(n), key=lambda i: scores[i])

    z_ok = True
    if mode in ("zscore", "both"):
        if n < 2:
            return None
        mean = sum(scores) / n
        std = (sum((x - mean) ** 2 for x in scores) / n) ** 0.5
        z_ok = std > 1e-9 and (scores[best] - mean) / std > k

    abs_ok = True
    if mode in ("absolute", "both"):
        abs_ok = floor is not None and scores[best] > floor

    return best if (z_ok and abs_ok) else None


@torch.no_grad()
def calibrate_floor(model: Any, tokenizer: Any, traces, device: str,
                    quantile: float = 0.99) -> float:
    """Absolute score floor = q-th percentile of genuine self-span scores on CLEAN
    traces. Calibrate on data held out from the eval seeds."""
    import numpy as np

    vals: list[float] = []
    for tr in traces:
        ids, spans, _asst = build_context(tokenizer, tr, device)
        vals.extend(span_scores(model, ids, spans))
    return float(np.quantile(vals, quantile)) if vals else float("inf")


def holed_mask(input_ids, start: int, end: int):
    """Attention mask with the flagged span's CONTENT tokens removed (true removal)."""
    m = torch.ones_like(input_ids)
    m[0, start:end] = 0
    return m


@torch.no_grad()
def generate_answer(
    model: Any, tokenizer: Any, input_ids, attention_mask, max_new_tokens: int
) -> str:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", 0)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad,
    )
    return tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)


def parse_answer(text: str, trace) -> int | None:
    """Which candidate does the free-text answer express? None = abstain."""
    t = text.lower()
    claim = trace.claim
    surfaces = claim.candidate_surfaces or []
    values = claim.candidate_values or []
    neg_idx = [i for i, s in enumerate(surfaces) if s == "negation"]
    if neg_idx and _NEG_RE.search(t):
        return neg_idx[0]
    # longest match first, so "the attic" beats a bare substring
    hits = [
        (len(v), i)
        for i, (v, s) in enumerate(zip(values, surfaces, strict=False))
        if s == "value" and v.lower() in t
    ]
    return max(hits)[1] if hits else None


def run_case(model: Any, tokenizer: Any, trace, cfg, condition: str,
             floor: float | None = None) -> dict:
    """One conversation under one condition; returns the graded outcome."""
    input_ids, spans, asst = build_context(tokenizer, trace, cfg.device)

    flagged = None
    if condition == "monitor_on":
        fi = flag_index(
            span_scores(model, input_ids, spans),
            k=cfg.k_threshold,
            floor=floor,
            mode=getattr(cfg, "flag_mode", "zscore"),
        )
        if fi is not None:
            flagged = asst[fi]
    elif condition == "oracle_excise":
        ci = trace.meta.get("corrupt_turn_index")
        flagged = next(((ti, s, e) for (ti, s, e) in asst if ti == ci), None)

    mask = (
        torch.ones_like(input_ids)
        if flagged is None
        else holed_mask(input_ids, flagged[1], flagged[2])
    )
    raw = generate_answer(model, tokenizer, input_ids, mask, cfg.max_new_tokens)
    text = strip_think(raw)

    parsed = parse_answer(text, trace)
    ci = trace.meta.get("corrupt_turn_index")
    return {
        "condition": condition,
        "answer": text.strip()[:120],
        "correct": (
            bool(parsed == trace.claim.correct_index) if parsed is not None else False
        ),
        "abstain": parsed is None,
        "excised": flagged is not None,
        "flagged_turn": None if flagged is None else flagged[0],
        "flag_hit": bool(flagged is not None and ci is not None and flagged[0] == ci),
    }
