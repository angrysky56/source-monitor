"""Tests for tokenizing trace with provenance boundaries."""

from __future__ import annotations

import pytest
import torch

from source_monitor.llm.task_render import Turn, Trace
from source_monitor.llm.provenance import (
    tokenize_with_provenance, check_prefix_stability, SpanAnnotation
)


class MockTokenizer:
    """Fake tokenizer simulating ChatML style formatting with return_offsets_mapping."""
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
        # A simple character-by-character tokenization to make offsets trivial.
        # Each char is a token. Offset of char i is (i, i+1).
        # We represent <|im_start|> and similar as individual characters for simplicity.
        n = len(text)
        input_ids = torch.arange(n).unsqueeze(0)
        offsets = torch.tensor([[i, i + 1] for i in range(n)]).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
        }


def test_prefix_stability_check():
    tokenizer = MockTokenizer()
    turns = [
        Turn(role="system", content="Sys", is_self=False, step_index=None),
        Turn(role="user", content="User command", is_self=False, step_index=0),
        Turn(role="assistant", content="Assistant emission", is_self=True, step_index=0, location_text="emission"),
    ]
    trace = Trace(
        turns=turns,
        query_object="object",
        ground_truth_final="final",
        op_kinds=[0],
        task=None,  # type: ignore
    )
    
    # Should pass without error
    check_prefix_stability(tokenizer, trace)


def test_tokenize_with_provenance_offsets():
    tokenizer = MockTokenizer()
    turns = [
        Turn(role="system", content="Sys", is_self=False, step_index=None),
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
    
    input_ids, spans = tokenize_with_provenance(tokenizer, trace)
    
    # 3 turns -> 3 spans
    assert len(spans) == 3
    
    # Assistant span should be the third one
    asst_span = spans[2]
    assert asst_span.kind == "assistant"
    assert asst_span.is_corrupted is False
    assert asst_span.claim_surface == "container"
    
    # The tokens for assistant turn should map to "The red ball is in box A."
    full_text = tokenizer.apply_chat_template([
        {"role": t.role, "content": t.content} for t in turns
    ])
    
    # Verify content slice
    start_char = full_text.find(turns[2].content)
    end_char = start_char + len(turns[2].content)
    content_slice = full_text[asst_span.start_token:asst_span.end_token]
    assert content_slice == turns[2].content
    
    # Verify location slice (A4)
    loc_start_char = full_text.find(turns[2].location_text, start_char)
    loc_end_char = loc_start_char + len(turns[2].location_text)
    loc_slice = full_text[asst_span.location_start_token:asst_span.location_end_token]
    assert loc_slice == turns[2].location_text
