"""Tests for the Phase 3 closed-loop monitor (CPU, no model)."""

from __future__ import annotations

import torch

from source_monitor.llm.loop.monitor import (
    build_context,
    flag_index,
    holed_mask,
    parse_answer,
)
from source_monitor.llm.ood import entity_prose


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, chat_turns, tokenize=False, add_generation_prompt=False):
        return "".join(
            f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in chat_turns
        )

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        n = len(text)
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "offset_mapping": torch.tensor([[i, i + 1] for i in range(n)]).unsqueeze(0),
        }


def test_flag_index_z_rule():
    assert flag_index([1.0, 1.0, 1.0, 1.0], 1.5) is None   # flat -> no flag
    assert flag_index([3.0], 1.5) is None                   # too few spans
    scores = [1.0, 1.0, 1.0, 9.0]
    assert flag_index(scores, 1.5) == 3                     # clear outlier flagged
    assert flag_index(scores, 5.0) is None                  # threshold above the outlier


def test_absolute_floor_keeps_the_monitor_quiet_when_nothing_is_wrong():
    """The relative z-rule fires on a quiet trace; the absolute floor must not."""
    quiet = [1.0, 1.05, 1.1, 1.4]  # nothing wrong, but the max is still ~1.7 sigma
    assert flag_index(quiet, k=1.5, mode="zscore") == 3  # the P-3.3 failure mode
    assert flag_index(quiet, floor=5.0, mode="absolute") is None
    assert flag_index(quiet, k=1.5, floor=5.0, mode="both") is None

    loud = [1.0, 1.0, 1.0, 9.0]  # a genuinely surprising span
    assert flag_index(loud, floor=5.0, mode="absolute") == 3
    assert flag_index(loud, floor=12.0, mode="absolute") is None
    assert flag_index(loud, k=1.5, floor=5.0, mode="both") == 3


def test_holed_mask_zeros_exactly_the_span():
    ids = torch.arange(10).unsqueeze(0)
    m = holed_mask(ids, 3, 6)
    assert m.shape == ids.shape
    assert m[0, 3:6].sum().item() == 0
    assert m[0, :3].sum().item() == 3
    assert m[0, 6:].sum().item() == 4


def test_parse_answer_values_negation_and_abstain():
    tr = entity_prose.generate(4, 1)[0]
    vals, surfs = tr.claim.candidate_values, tr.claim.candidate_surfaces
    vi = surfs.index("value")
    assert parse_answer(f"I believe it is in {vals[vi]} right now.", tr) == vi
    ni = surfs.index("negation")
    assert parse_answer("It isn't anywhere anymore.", tr) == ni
    # Real model phrasings (regression: a literal cue list missed these)
    assert parse_answer("The blue mug is not currently anywhere.", tr) == ni
    assert parse_answer("It is no longer anywhere in the house.", tr) == ni
    assert parse_answer("The mug is nowhere.", tr) == ni
    assert parse_answer("Bananas and telescopes.", tr) is None  # abstain


def test_strip_think_hides_reasoning_from_the_grader():
    from source_monitor.llm.loop.monitor import strip_think

    assert strip_think("<think>maybe the cellar</think>It is in the attic.") == "It is in the attic."
    # Unterminated think block (budget ran out mid-reasoning): the reasoning names
    # locations, so NOTHING may survive to be graded.
    assert strip_think("<think>Let me check. It was moved to the cellar") == ""
    assert strip_think("It is in the attic.") == "It is in the attic."


def test_build_context_excludes_claim_and_spans_are_valid():
    tok = _CharTokenizer()
    tr = entity_prose.generate(2, 1)[0]
    ids, spans, asst = build_context(tok, tr, "cpu")
    L = int(ids.shape[1])
    assert asst, "expected assistant emissions in context"
    for (ti, s, e) in asst:
        assert 0 <= s < e <= L
        assert ti < tr.claim.turn_index  # context stops before the final claim
    # one span per assistant turn preceding the claim
    n_asst_before = sum(
        1 for t in tr.turns[: tr.claim.turn_index] if t.role == "assistant"
    )
    assert len(asst) == n_asst_before
