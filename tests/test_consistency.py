"""CPU tests for the F26 query-paraphrase family (pure Python, no model)."""

from __future__ import annotations

from source_monitor.llm.loop.consistency import (
    QFRAMES,
    _frames,
    _question_turn_index,
    paraphrase_query,
)
from source_monitor.llm.ood import factual_qa


def _a_factual_trace():
    return factual_qa.generate(1, 20)[0]


def test_question_turn_is_last_user_before_claim() -> None:
    tr = _a_factual_trace()
    qi = _question_turn_index(tr)
    assert tr.turns[qi].role == "user"
    assert qi == tr.claim.turn_index - 1  # question immediately precedes the claim


def test_identity_frame_is_a_noop() -> None:
    tr = _a_factual_trace()
    out = paraphrase_query(tr, QFRAMES[0])  # "{q}"
    assert [t.content for t in out.turns] == [t.content for t in tr.turns]


def test_frame_wraps_only_the_question() -> None:
    tr = _a_factual_trace()
    qi = _question_turn_index(tr)
    q = tr.turns[qi].content
    out = paraphrase_query(tr, "Please answer: {q}")
    assert out.turns[qi].content == f"Please answer: {q}"
    # every other turn is byte-identical
    for i, (a, b) in enumerate(zip(tr.turns, out.turns, strict=True)):
        if i != qi:
            assert a.content == b.content


def test_paraphrase_is_nondestructive() -> None:
    tr = _a_factual_trace()
    before = [t.content for t in tr.turns]
    paraphrase_query(tr, "Quick question: {q}")
    assert [t.content for t in tr.turns] == before


def test_frames_deterministic_and_include_identity() -> None:
    assert _frames(3) == list(QFRAMES[:3])
    assert _frames(3)[0] == "{q}"  # control always first
    assert len(_frames(6)) == 6
    assert len(_frames(10)) == 10  # wraps past the pool without error


def test_candidate_machinery_survives_paraphrase() -> None:
    # The claim/candidate set must be untouched so make_variant still works.
    tr = _a_factual_trace()
    out = paraphrase_query(tr, "I'd like to know: {q}")
    assert out.claim.candidate_contents == tr.claim.candidate_contents
    assert out.claim.correct_index == tr.claim.correct_index
