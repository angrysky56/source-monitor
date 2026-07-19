"""Tests for retrospective surprisal computation."""

from __future__ import annotations

import pytest
import torch

from source_monitor.llm.provenance import SpanAnnotation
from source_monitor.llm.telemetry import retrospective_surprisal, SpanScore


class MockModelOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class MockModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.Tensor) -> MockModelOutput:
        # Return deterministic/simple logits
        # Shape: (1, L, V)
        L = input_ids.shape[1]
        logits = torch.zeros((1, L, self.vocab_size), dtype=torch.float32)
        # For even positions, let's make token 0 extremely probable
        # For odd positions, let's make token 1 extremely probable
        for i in range(L):
            target_tok = i % 2
            logits[0, i, target_tok] = 20.0  # high logit -> logp near 0
        return MockModelOutput(logits)


def test_retrospective_surprisal_calculation():
    vocab_size = 10
    model = MockModel(vocab_size)
    
    # Trace input_ids: 6 tokens
    # Predictable tokens: input_ids matches position modulo 2
    # Unpredictable tokens: input_ids does not match
    # Since position pos is predicted by logits at pos - 1:
    # - pos 1 (pred 0): input_ids[1] = 0 (predictable)
    # - pos 2 (pred 1): input_ids[2] = 1 (predictable)
    # - pos 3 (pred 0): input_ids[3] = 9 (unpredictable)
    # - pos 4 (pred 1): input_ids[4] = 1 (predictable)
    # - pos 5 (pred 0): input_ids[5] = 0 (predictable)
    input_ids = torch.tensor([[0, 0, 1, 9, 1, 0]], dtype=torch.long)
    
    # Span 1 (predictable): tokens [0, 3) -> content is [0, 1, 0]
    # Span 2 (unpredictable at index 3): tokens [3, 6) -> content is [9, 0, 1]
    # Note: token at index 3 is 9, but odd position predicts 1. So token 9 has very high surprisal.
    spans = [
        SpanAnnotation(
            start_token=1,  # exclude token 0 as we can't predict it
            end_token=3,
            kind="assistant",
            step_index=0,
            is_corrupted=False,
            claim_surface="container",
            location_start_token=2,
            location_end_token=3,
        ),
        SpanAnnotation(
            start_token=3,
            end_token=6,
            kind="assistant",
            step_index=1,
            is_corrupted=True,
            claim_surface="nowhere",
            location_start_token=3,
            location_end_token=4,
        ),
    ]
    
    scores = retrospective_surprisal(model, input_ids, spans)
    
    assert len(scores) == 2
    
    s0, s1 = scores
    assert s0.step_index == 0
    assert s0.is_corrupted is False
    assert s0.claim_surface == "container"
    # s0 is highly predictable, so surprisal should be very low
    assert s0.mean_neglogp < 1.0
    
    # s1 has token 9 at position 3, which is highly unpredictable
    # So max_neglogp and mean_neglogp should be high
    assert s1.step_index == 1
    assert s1.is_corrupted is True
    assert s1.claim_surface == "nowhere"
    assert s1.max_neglogp > 10.0
    # slot_only_neglogp checks tokens [3, 4) which is token 9. So it should be high.
    assert s1.slot_only_neglogp > 10.0


def test_contrastive_slot_scores():
    from source_monitor.llm.task_render import Turn, Trace
    from source_monitor.llm.telemetry import contrastive_slot_scores
    
    class MockTokenizer:
        def apply_chat_template(
            self, chat_turns, tokenize=False, add_generation_prompt=False
        ) -> str:
            res = ""
            for turn in chat_turns:
                res += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
            return res

        def __call__(
            self, text: str, return_offsets_mapping=False, return_tensors=None
        ) -> dict:
            n = len(text)
            input_ids = torch.zeros((1, n), dtype=torch.long)
            offsets = torch.tensor([[i, i + 1] for i in range(n)]).unsqueeze(0)
            return {
                "input_ids": input_ids,
                "offset_mapping": offsets,
            }

    class MockModelOutput:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class MockModel:
        def __call__(self, input_ids: torch.Tensor, *args, **kwargs) -> MockModelOutput:
            L = input_ids.shape[1]
            logits = torch.zeros((input_ids.shape[0], L, 50000), dtype=torch.float32)
            return MockModelOutput(logits)

    tokenizer = MockTokenizer()
    model = MockModel()
    
    turns = [
        Turn(role="system", content="Sys instruction", is_self=False, step_index=None),
        Turn(role="user", content="Command", is_self=False, step_index=0),
        Turn(role="assistant", content="The red ball is in box A.", is_self=True, step_index=0, claim_surface="container", location_text="box A"),
    ]
    trace = Trace(
        turns=turns,
        query_object="red ball",
        ground_truth_final="box A",
        op_kinds=[0],
        task=None,  # type: ignore
    )
    
    scores = contrastive_slot_scores(model, tokenizer, trace, device="cpu")

    assert len(scores) == 1
    s0 = scores[0]
    assert s0.step_index == 0
    assert s0.claim_surface == "container"
    # Contrastive surprisal should be a valid float >= 0
    assert s0.mean_neglogp >= 0.0
    assert s0.slot_only_neglogp >= 0.0


class _CharTokenizer:
    """ChatML-style tokenizer where every character is its own token.

    Mirrors the fake used in test_provenance_llm so span math is trivial to
    verify: char i has offset (i, i+1).
    """

    def apply_chat_template(
        self, chat_turns, tokenize=False, add_generation_prompt=False
    ) -> str:
        res = ""
        for turn in chat_turns:
            res += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
        return res

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        n = len(text)
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "offset_mapping": torch.tensor([[i, i + 1] for i in range(n)]).unsqueeze(0),
        }


def test_render_chatml_matches_apply_chat_template():
    """The Python ChatML render must be byte-identical to the template render.

    This is the correctness guarantee for the Phase 0b speedup: the contrastive
    scorer builds the ChatML string in Python instead of re-invoking Jinja per
    candidate. If the two ever diverge, span indices become silently wrong.
    """
    from source_monitor.llm.task_render import Turn
    from source_monitor.llm.telemetry import render_chatml

    tok = _CharTokenizer()
    turns = [
        Turn(role="system", content="Track objects.", is_self=False, step_index=None),
        Turn(role="user", content="Put the red ball in box A.", is_self=False, step_index=0),
        Turn(role="assistant", content="The red ball is in box A.", is_self=True, step_index=0),
    ]
    py_render = render_chatml(turns)
    template_render = tok.apply_chat_template(
        [{"role": t.role, "content": t.content} for t in turns],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert py_render == template_render


def test_encode_candidate_spans_align():
    """_encode_candidate must recover the exact content and slot character spans."""
    from source_monitor.llm.telemetry import _encode_candidate, render_chatml
    from source_monitor.llm.task_render import Turn

    tok = _CharTokenizer()
    prefix_turns = [
        Turn(role="system", content="Track objects.", is_self=False, step_index=None),
        Turn(role="user", content="Put the red ball in box A.", is_self=False, step_index=0),
    ]
    prefix_text = render_chatml(prefix_turns)
    cand_content = "The red ball is in box A."
    location_text = "box A"

    ids, content_pos, slot_pos = _encode_candidate(
        tok, prefix_text, cand_content, location_text
    )
    full_text = prefix_text + "<|im_start|>assistant\n" + cand_content + "<|im_end|>\n"

    # With char-per-token offsets, positions index directly into full_text.
    content_slice = full_text[content_pos[0] : content_pos[-1] + 1]
    assert content_slice == cand_content
    slot_slice = full_text[slot_pos[0] : slot_pos[-1] + 1]
    assert slot_slice == location_text


def test_contrastive_candidate_set_is_trace_local():
    """Candidate set = containers named in THIS trace + nowhere, not the global pool."""
    from source_monitor.llm.task_render import Turn, Trace
    from source_monitor.llm.telemetry import contrastive_slot_scores

    class MockModelOutput:
        def __init__(self, logits):
            self.logits = logits

    class MockModel:
        def __call__(self, input_ids, *args, **kwargs):
            L = input_ids.shape[1]
            return MockModelOutput(
                torch.zeros((input_ids.shape[0], L, 5000), dtype=torch.float32)
            )

    # Trace only ever names box A and box B (never C..H).
    turns = [
        Turn(role="system", content="Track objects.", is_self=False, step_index=None),
        Turn(role="user", content="Put the red ball in box A.", is_self=False, step_index=0),
        Turn(role="assistant", content="The red ball is in box A.", is_self=True,
             step_index=0, claim_surface="container", location_text="box A"),
        Turn(role="user", content="Move the red ball to box B.", is_self=False, step_index=1),
        Turn(role="assistant", content="The red ball is in box B.", is_self=True,
             step_index=1, claim_surface="container", location_text="box B"),
    ]
    trace = Trace(
        turns=turns, query_object="red ball", ground_truth_final="box B",
        op_kinds=[0, 0], task=None,  # type: ignore
    )

    # Uniform logits -> each score is log(|C|). Only box A, box B, nowhere are in
    # play, so |C| = 3 and the score is log(3), not log(9).
    import math
    scores = contrastive_slot_scores(MockModel(), _CharTokenizer(), trace, device="cpu")
    assert len(scores) == 2
    for s in scores:
        assert s.mean_neglogp == pytest.approx(math.log(3), abs=1e-5)
