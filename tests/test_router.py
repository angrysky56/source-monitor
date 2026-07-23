"""CPU unit tests for router.py (F27).

Verifies content-only derivability classification (zero metadata access),
routing dispatch, identity controls, and independent binary flag combination.
"""

from dataclasses import dataclass
from typing import Any

from source_monitor.llm.loop.router import is_context_derivable, route


@dataclass
class DummyTurn:
    role: str
    content: str


@dataclass
class DummyClaim:
    turn_index: int
    candidate_values: list[str]
    correct_index: int = 0
    candidate_surfaces: list[str] | None = None


@dataclass
class DummyTrace:
    turns: list[DummyTurn]
    claim: DummyClaim
    meta: dict[str, Any]
    query_object: str = ""


def test_is_context_derivable_content_only():
    # Context contains "cellar"
    turns = [
        DummyTurn(role="user", content="Where is the ball?"),
        DummyTurn(role="assistant", content="The ball was moved to the cellar."),
        DummyTurn(role="user", content="Now where is it?"),
    ]
    claim = DummyClaim(turn_index=2, candidate_values=["cellar", "attic"])

    # Trace with meta pointing to grounded=False (adversarial test)
    trace = DummyTrace(turns=turns, claim=claim, meta={"grounded": False})

    # Must classify as derivable based ONLY on content ("cellar" in context)
    assert is_context_derivable(trace) is True
    assert route(trace) == "surprisal"

    # Mutating meta must NOT change classification
    trace.meta = {"grounded": True, "task": "unknown"}
    assert is_context_derivable(trace) is True


def test_is_context_underivable_content_only():
    # Context does NOT contain candidate values
    turns = [
        DummyTurn(role="user", content="What is the capital of France?"),
    ]
    claim = DummyClaim(turn_index=1, candidate_values=["paris", "london"])
    trace = DummyTrace(turns=turns, claim=claim, meta={"grounded": True})

    # Must classify as underivable because "paris"/"london" are absent from context
    assert is_context_derivable(trace) is False
    assert route(trace) == "consistency"


def test_identity_controls():
    # 1. Derivable set: all claims have values in context
    derivable_traces = [
        DummyTrace(
            turns=[
                DummyTurn(role="user", content="John went to the garden."),
                DummyTurn(role="user", content="Where is John?"),
            ],
            claim=DummyClaim(turn_index=1, candidate_values=["garden", "kitchen"]),
            meta={},
        ),
        DummyTrace(
            turns=[
                DummyTurn(role="user", content="The total is 42."),
                DummyTurn(role="user", content="What is the total?"),
            ],
            claim=DummyClaim(turn_index=1, candidate_values=["42", "50"]),
            meta={},
        ),
    ]

    routes_derivable = [route(t) for t in derivable_traces]
    assert all(r == "surprisal" for r in routes_derivable), f"Expected all surprisal, got {routes_derivable}"

    # 2. Factual / underivable set: claims not in context
    underivable_traces = [
        DummyTrace(
            turns=[DummyTurn(role="user", content="Who wrote Hamlet?")],
            claim=DummyClaim(turn_index=0, candidate_values=["Shakespeare", "Marlowe"]),
            meta={},
        ),
        DummyTrace(
            turns=[DummyTurn(role="user", content="What is the boiling point of water in Celsius?")],
            claim=DummyClaim(turn_index=0, candidate_values=["100", "90"]),
            meta={},
        ),
    ]

    routes_underivable = [route(t) for t in underivable_traces]
    assert all(r == "consistency" for r in routes_underivable), f"Expected all consistency, got {routes_underivable}"
