"""Tests for text-level corruption injection."""

from __future__ import annotations

import random
import pytest

from source_monitor.task import generate_task, REMOVE, NOWHERE
from source_monitor.llm.task_render import render_trace
from source_monitor.llm.corruption import (
    inject_ghost_text, inject_mislocation_text, inject_phantom_text,
    inject_all_types, CorruptionRecord
)


def _make_trace(seed=42, n_ops=8, remove_prob=0.4):
    rng = random.Random(seed)
    return render_trace(generate_task(rng, n_ops=n_ops, remove_prob=remove_prob))


def test_ghost_corrupts_removal_step():
    trace = _make_trace()
    rng = random.Random(99)
    rec = inject_ghost_text(trace, rng)
    if rec is not None:
        assert rec.corruption_type == "ghost"
        assert rec.original_surface == "nowhere"
        assert rec.corrupted_surface == "container"
        assert "nowhere" not in rec.corrupted_content.lower()
        assert "box" in rec.corrupted_content.lower()
        
        # The corrupted turn is marked
        corrupted_turn = rec.trace.turns[2 * (rec.step_index + 1)]
        assert corrupted_turn.is_corrupted
        assert corrupted_turn.claim_surface == "container"


def test_mislocation_changes_container():
    trace = _make_trace()
    rng = random.Random(99)
    rec = inject_mislocation_text(trace, rng)
    if rec is not None:
        assert rec.corruption_type == "mislocation"
        assert rec.original_surface == "container"
        assert rec.corrupted_surface == "container"
        assert rec.corrupted_content != rec.original_content


def test_phantom_says_nowhere():
    trace = _make_trace()
    rng = random.Random(99)
    rec = inject_phantom_text(trace, rng)
    if rec is not None:
        assert rec.corruption_type == "phantom"
        assert rec.original_surface == "container"
        assert rec.corrupted_surface == "nowhere"
        assert "nowhere" in rec.corrupted_content.lower()


def test_corruption_preserves_other_turns():
    trace = _make_trace()
    rng = random.Random(99)
    rec = inject_ghost_text(trace, rng)
    if rec is None:
        pytest.skip("No eligible ghost step in this trace")
    for i, (orig, corrupt) in enumerate(zip(trace.turns, rec.trace.turns)):
        if i != 2 * (rec.step_index + 1):  # not the corrupted position
            assert orig.content == corrupt.content
            assert corrupt.is_corrupted is False


def test_returns_none_when_ineligible():
    # Task with zero removes -> no ghost possible
    rng = random.Random(42)
    task = generate_task(rng, n_ops=4, remove_prob=0.0)
    trace = render_trace(task)
    assert inject_ghost_text(trace, random.Random(0)) is None


def test_inject_all_types_returns_dict():
    trace = _make_trace(seed=7, remove_prob=0.3)
    rng = random.Random(42)
    results = inject_all_types(trace, rng)
    assert set(results.keys()) == {"ghost", "mislocation", "phantom"}
    for key, val in results.items():
        assert val is None or isinstance(val, CorruptionRecord)


def test_corruption_does_not_mutate_original():
    trace = _make_trace()
    original_contents = [t.content for t in trace.turns]
    rng = random.Random(99)
    inject_ghost_text(trace, rng)
    for orig, turn in zip(original_contents, trace.turns):
        assert turn.content == orig
        assert turn.is_corrupted is False
