"""Layer 0: Provenance bookkeeping.

Tokenizes a Trace and extracts the precise token spans for each assistant turn
content (excluding chat template boilerplate) and the location-slot tokens (A4).

A1: Validates prefix stability of the chat template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from source_monitor.llm.task_render import Trace, Turn


@dataclass
class SpanAnnotation:
    """Token-level annotation for a turn's content."""
    start_token: int              # inclusive token index in full sequence
    end_token: int                # exclusive token index
    kind: str                     # "system" | "user" | "assistant"
    step_index: int | None
    is_corrupted: bool
    claim_surface: str | None = None
    # A4: Location slot token boundaries (relative to the full token sequence)
    location_start_token: int | None = None
    location_end_token: int | None = None


def check_prefix_stability(tokenizer: Any, trace: Trace) -> None:
    """A1: Verify that rendering prefixes of turns is strictly append-only.

    Raises ValueError if prefix stability is violated.
    """
    # System turn + pairs of (user, assistant)
    turns = trace.turns
    prev_render = ""
    for i in range(1, len(turns) + 1):
        sub_turns = [
            {"role": t.role, "content": t.content}
            for t in turns[:i]
        ]
        curr_render = tokenizer.apply_chat_template(
            sub_turns,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not curr_render.startswith(prev_render):
            # Template violated prefix stability!
            raise ValueError(
                f"Prefix stability violated at turn {i-1} ({turns[i-1].role}).\n"
                f"Prev: {prev_render!r}\n"
                f"Curr: {curr_render!r}"
            )
        prev_render = curr_render


def tokenize_with_provenance(
    tokenizer: Any,
    trace: Trace,
    device: str = "cpu",
) -> tuple[Tensor, list[SpanAnnotation]]:
    """Tokenize the trace and find exact token spans for each assistant turn content.

    A1: Verifies prefix stability, filters out chat template boilerplate.
    A4: Returns location slot boundaries inside each assistant turn.
    """
    # Convert Trace turns to HF chat template list of dicts
    chat_turns = [
        {"role": t.role, "content": t.content}
        for t in trace.turns
    ]
    
    # Render full text
    full_text = tokenizer.apply_chat_template(
        chat_turns,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Tokenize and get offset mappings
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    offsets = encoding["offset_mapping"][0].tolist()  # list of (start, end) char index
    
    # Check prefix stability (fails closed if violated)
    check_prefix_stability(tokenizer, trace)
    
    # Locate each turn's content and location text within full_text
    spans: list[SpanAnnotation] = []
    search_pos = 0
    
    for turn in trace.turns:
        # Find this turn's content in full_text
        start_char = full_text.find(turn.content, search_pos)
        if start_char == -1:
            raise ValueError(f"Could not find turn content {turn.content!r} in full text.")
        end_char = start_char + len(turn.content)
        search_pos = end_char
        
        # Map character offsets to tokens
        content_tokens = []
        for t_idx, (tok_start, tok_end) in enumerate(offsets):
            # Special case for BOS/EOS or zero-length tokens (like special markers)
            if tok_start == tok_end == 0:
                continue
            # Token is inside the content range
            if tok_start >= start_char and tok_end <= end_char:
                content_tokens.append(t_idx)
                
        if not content_tokens:
            # Empty content (like system prompt or empty turns)
            start_tok = end_tok = 0
        else:
            start_tok = content_tokens[0]
            # exclusive end
            end_tok = content_tokens[-1] + 1
            
        # Find location text boundaries (A4) if present
        loc_start_tok = None
        loc_end_tok = None
        if turn.is_self and turn.location_text is not None:
            loc_idx = turn.content.find(turn.location_text)
            if loc_idx != -1:
                loc_start_char = start_char + loc_idx
                loc_end_char = loc_start_char + len(turn.location_text)
                
                loc_tokens = []
                for t_idx in content_tokens:
                    tok_start, tok_end = offsets[t_idx]
                    if tok_start >= loc_start_char and tok_end <= loc_end_char:
                        loc_tokens.append(t_idx)
                if loc_tokens:
                    loc_start_tok = loc_tokens[0]
                    loc_end_tok = loc_tokens[-1] + 1
                    
        spans.append(
            SpanAnnotation(
                start_token=start_tok,
                end_token=end_tok,
                kind=turn.role,
                step_index=turn.step_index,
                is_corrupted=turn.is_corrupted,
                claim_surface=turn.claim_surface,
                location_start_token=loc_start_tok,
                location_end_token=loc_end_tok,
            )
        )
        
    return input_ids, spans
