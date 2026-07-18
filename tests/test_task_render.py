"""Tests for task rendering to chat traces."""

from __future__ import annotations

import random
import pytest

from source_monitor.task import generate_task, REMOVE, NOWHERE
from source_monitor.llm.task_render import (
    render_trace, obj_name, con_name, loc_name, Turn, Trace,
    OBJECT_NAMES, CONTAINER_NAMES, SYSTEM_PROMPT
)


def test_render_produces_valid_turn_sequence():
    """System, then alternating user/assistant for each step."""
    rng = random.Random(42)
    task = generate_task(rng, n_ops=4)
    trace = render_trace(task)
    assert trace.turns[0].role == "system"
    for i in range(1, len(trace.turns), 2):
        assert trace.turns[i].role == "user"
        assert trace.turns[i + 1].role == "assistant"


def test_render_deterministic():
    """Same seed -> same text."""
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    t1 = render_trace(generate_task(rng1, n_ops=4))
    t2 = render_trace(generate_task(rng2, n_ops=4))
    for a, b in zip(t1.turns, t2.turns):
        assert a.content == b.content


def test_self_positions_are_assistant_turns():
    rng = random.Random(42)
    trace = render_trace(generate_task(rng, n_ops=6))
    for turn in trace.turns:
        assert turn.is_self == (turn.role == "assistant")


def test_claim_surface_types():
    """Every assistant turn has claim_surface in {'container', 'nowhere'}."""
    rng = random.Random(42)
    # Use a task likely to have both types
    task = generate_task(rng, n_ops=8, remove_prob=0.4)
    trace = render_trace(task)
    for turn in trace.turns:
        if turn.is_self:
            assert turn.claim_surface in ("container", "nowhere")
        else:
            assert turn.claim_surface is None


def test_location_text_present_in_content():
    """A4: location_text appears verbatim in the assistant content."""
    rng = random.Random(42)
    task = generate_task(rng, n_ops=8)
    trace = render_trace(task)
    for turn in trace.turns:
        if turn.is_self:
            assert turn.location_text is not None
            assert turn.location_text in turn.content


def test_correct_step_count():
    rng = random.Random(42)
    n_ops = 5
    task = generate_task(rng, n_ops=n_ops)
    trace = render_trace(task)
    # 1 system + n_ops * 2 (user+assistant)
    assert len(trace.turns) == 1 + n_ops * 2


def test_object_name_consistency():
    """The query object name appears in every assistant turn."""
    rng = random.Random(42)
    task = generate_task(rng, n_ops=4)
    trace = render_trace(task)
    for turn in trace.turns:
        if turn.is_self:
            assert trace.query_object in turn.content
