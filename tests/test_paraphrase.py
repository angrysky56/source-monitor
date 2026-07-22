"""CPU tests for the F24 paraphrase family generator (pure Python, no model).

These guard the two invariants the whole experiment rests on: family members are
FACT-FAITHFUL (same object, same per-turn location, same corruption) and
SPAN-INVARIANT (the held turn, claim, system, and query are byte-identical). The
scoring reduction (identity -> std 0) is validated end-to-end by the runner's
control gate, which needs a model.
"""

from __future__ import annotations

import random

from source_monitor.llm.loop.paraphrase import (
    _is_removal,
    _query_object,
    rerender,
    rerender_arithmetic,
)
from source_monitor.llm.ood import arithmetic, entity_prose
from source_monitor.llm.task_render import Turn


def _facts(trace):
    """The fact fingerprint that a faithful paraphrase must preserve exactly."""
    return [(t.role, t.location_text, t.is_corrupted, t.step_index) for t in trace.turns]


def _a_trace(corrupt: bool = False):
    traces = entity_prose.generate(7, 20, corrupt_mid=corrupt)
    if corrupt:
        traces = [t for t in traces if t.meta.get("corrupt_turn_index") is not None]
    return traces[0]


def test_query_object_parse() -> None:
    tr = _a_trace()
    obj = _query_object(tr)
    assert obj is not None
    # every turn mentions the object, so it must appear in the first user turn
    first_user = next(t for t in tr.turns if t.role == "user")
    assert obj in first_user.content


def test_is_removal() -> None:
    assert _is_removal(Turn("user", "Take the red ball away.", False, 2))
    assert _is_removal(Turn("assistant", "Okay, the red ball has been removed.", True, 2))
    assert not _is_removal(Turn("user", "Now move the red ball to the attic.", False, 2))


def test_rerender_preserves_facts() -> None:
    tr = _a_trace()
    out = rerender(tr, hold_turn_index=2, rng=random.Random(1))
    assert _facts(out) == _facts(tr)  # object, locations, corruption, steps all held
    assert len(out.turns) == len(tr.turns)


def test_rerender_holds_pinned_turns_byte_identical() -> None:
    tr = _a_trace()
    hold = 2
    out = rerender(tr, hold_turn_index=hold, rng=random.Random(3))
    claim_ti = tr.claim.turn_index
    for i, (a, b) in enumerate(zip(tr.turns, out.turns, strict=True)):
        pinned = (
            i == hold
            or i == claim_ti
            or a.role == "system"
            or (a.role == "user" and a.content.startswith("Where is the "))
            or _is_removal(a)
        )
        if pinned:
            assert a.content == b.content, f"turn {i} should be byte-identical"


def test_rerender_actually_rewords_something() -> None:
    tr = _a_trace()
    # Across several seeds, at least one support turn must get a new surface form.
    changed = False
    for s in range(8):
        out = rerender(tr, hold_turn_index=2, rng=random.Random(s))
        if any(a.content != b.content for a, b in zip(tr.turns, out.turns, strict=True)):
            changed = True
            break
    assert changed


def test_rerender_deterministic_and_nondestructive() -> None:
    tr = _a_trace()
    before = [t.content for t in tr.turns]
    a = rerender(tr, 2, random.Random(42))
    b = rerender(tr, 2, random.Random(42))
    assert [t.content for t in a.turns] == [t.content for t in b.turns]
    assert [t.content for t in tr.turns] == before  # original untouched


def test_corrupt_turn_stays_a_lie_when_reworded() -> None:
    tr = _a_trace(corrupt=True)
    ci = tr.meta["corrupt_turn_index"]
    # Hold a DIFFERENT span so the corrupt ack itself gets reworded.
    hold = next(i for i, t in enumerate(tr.turns)
                if t.role == "assistant" and i != ci and i != tr.claim.turn_index)
    out = rerender(tr, hold_turn_index=hold, rng=random.Random(5))
    assert out.turns[ci].is_corrupted is True
    assert out.turns[ci].location_text == tr.turns[ci].location_text  # same wrong loc
    assert out.turns[ci].location_text in out.turns[ci].content  # still asserts it


def test_held_corrupt_turn_is_byte_identical() -> None:
    tr = _a_trace(corrupt=True)
    ci = tr.meta["corrupt_turn_index"]
    out = rerender(tr, hold_turn_index=ci, rng=random.Random(9))
    assert out.turns[ci].content == tr.turns[ci].content


# --- arithmetic (the F24d long-ξ hard task) ---------------------------------- #

def _an_arith_trace(corrupt: bool = False):
    trs = arithmetic.generate(11, 20, corrupt_mid=corrupt)
    if corrupt:
        trs = [t for t in trs if t.meta.get("corrupt_turn_index") is not None]
    return trs[0]


def test_arithmetic_corrupt_mid_plants_a_lie() -> None:
    tr = _an_arith_trace(corrupt=True)
    ci = tr.meta["corrupt_turn_index"]
    assert ci is not None
    assert tr.turns[ci].is_corrupted is True
    assert tr.turns[ci].step_index not in (0, None)  # never the first total
    # the final claim stays CORRECT (lie is mid-context, must be re-derived past)
    assert tr.claim.correct_index == tr.claim.emitted_index


def test_arithmetic_clean_mid_spans_are_annotated() -> None:
    # Consistent scoring needs location_text on mid totals for planted AND clean.
    tr = _an_arith_trace(corrupt=False)
    mids = [t for t in tr.turns if t.role == "assistant" and t.step_index not in (0, None)]
    assert mids and all(t.location_text is not None for t in mids)


def test_arithmetic_rerender_preserves_facts_and_holds_numbers() -> None:
    tr = _an_arith_trace(corrupt=True)
    out = rerender_arithmetic(tr, hold_turn_index=3, rng=random.Random(2))
    assert _facts(out) == _facts(tr)  # numbers, corruption, steps preserved
    # every reworded total still literally states its (held) number
    for a, b in zip(tr.turns, out.turns, strict=True):
        if a.location_text is not None:
            assert b.location_text in b.content


def test_arithmetic_corrupt_total_stays_wrong_when_reworded() -> None:
    tr = _an_arith_trace(corrupt=True)
    ci = tr.meta["corrupt_turn_index"]
    hold = next(i for i, t in enumerate(tr.turns)
                if t.role == "assistant" and i != ci and i != tr.claim.turn_index)
    out = rerender_arithmetic(tr, hold_turn_index=hold, rng=random.Random(4))
    assert out.turns[ci].is_corrupted is True
    assert out.turns[ci].location_text == tr.turns[ci].location_text
    assert out.turns[ci].location_text in out.turns[ci].content


def test_arithmetic_default_generate_unchanged_without_corrupt() -> None:
    # corrupt_mid defaults off; no lie planted, so F19 arithmetic behaviour holds.
    trs = arithmetic.generate(3, 10)
    assert all(t.meta.get("corrupt_turn_index") is None for t in trs)
    assert all(not any(x.is_corrupted for x in t.turns) for t in trs)
