"""Domain-agnostic trace/claim abstraction + generalized scorers for Phase 1.

The Phase 0 scorers are entity-tracking-specific (candidate set = containers ∪
"nowhere", claim rendered via fixed templates). Phase 1 generalizes this: each
domain supplies, for the single corruptible claim in a trace, the FULL set of
candidate assistant-turn contents (so any structural variation — "The total is
42." vs "There is no whole-number total." — is owned by the domain, not the
scorer). Raw scoring reuses provenance + retrospective_surprisal unchanged.

Conventions:
- The corruptible claim is always the FINAL assistant turn of the trace, so raw
  scoring reads scores[-1].
- Candidates carry a parallel `value` substring for the value-only aggregation
  (analogous to Phase 0's location slot) and a `surface` tag
  ("value" | "negation") for matched-surface stratification (A2).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import torch

from source_monitor.llm.provenance import tokenize_with_provenance
from source_monitor.llm.task_render import Trace, Turn
from source_monitor.llm.telemetry import (
    _encode_candidate,
    _mean_lp,
    logsumexp,
    render_chatml,
    retrospective_surprisal,
)


@dataclass
class OODClaim:
    """The single corruptible self-claim in an OOD trace (the final asst turn).

    candidate_contents / candidate_values / candidate_surfaces are parallel lists
    describing the full enumerable alternative set for this claim slot. They are
    None for free-text domains (e.g. code), where only raw scoring applies.
    """

    turn_index: int
    correct_index: int = 0
    emitted_index: int = 0
    candidate_contents: list[str] | None = None
    candidate_values: list[str] | None = None
    candidate_surfaces: list[str] | None = None
    # Free-text claims (candidate_contents is None) track corruption directly:
    free_is_corrupted: bool = False
    free_surface: str = "value"

    @property
    def surface_type(self) -> str:
        if self.candidate_surfaces is None:
            return self.free_surface
        return self.candidate_surfaces[self.emitted_index]

    @property
    def is_corrupted(self) -> bool:
        if self.candidate_contents is None:
            return self.free_is_corrupted
        return self.emitted_index != self.correct_index


@dataclass
class OODTrace:
    domain: str
    turns: list[Turn]
    claim: OODClaim
    meta: dict[str, Any] = field(default_factory=dict)

    def as_trace(self) -> Trace:
        """Wrap for provenance/telemetry (which only read `.turns`)."""
        return Trace(
            turns=self.turns,
            query_object="",
            ground_truth_final="",
            op_kinds=[],
            task=None,  # type: ignore[arg-type]
        )


@dataclass
class ClaimScore:
    """Detection scores for one claim under one scorer, all three aggregations."""

    surface_type: str
    is_corrupted: bool
    mean_neglogp: float
    max_neglogp: float
    value_only_neglogp: float


def make_variant(trace: OODTrace, emitted_index: int) -> OODTrace:
    """Return a copy of `trace` whose final claim emits candidate `emitted_index`.

    Updates the final assistant turn's content/location_text/is_corrupted and the
    claim's emitted_index. Requires enumerable candidates.
    """
    claim = trace.claim
    assert claim.candidate_contents is not None, "make_variant needs candidates"
    new = copy.deepcopy(trace)
    turn = new.turns[claim.turn_index]
    turn.content = claim.candidate_contents[emitted_index]
    turn.location_text = claim.candidate_values[emitted_index]
    turn.is_corrupted = emitted_index != claim.correct_index
    turn.claim_surface = claim.candidate_surfaces[emitted_index]
    new.claim.emitted_index = emitted_index
    return new


def make_free_variant(
    trace: OODTrace, content: str, *, is_corrupted: bool, surface: str = "value"
) -> OODTrace:
    """Variant for free-text claims (no enumerable candidates), e.g. code_trace.

    Directly sets the final claim turn's content and the corruption flag.
    """
    new = copy.deepcopy(trace)
    turn = new.turns[new.claim.turn_index]
    turn.content = content
    turn.is_corrupted = is_corrupted
    turn.claim_surface = surface
    new.claim.free_is_corrupted = is_corrupted
    new.claim.free_surface = surface
    return new


def corrupt_to_value(trace: OODTrace, rng: random.Random) -> OODTrace | None:
    """Corrupt the claim to a WRONG value-surface candidate (misloc-analog)."""
    claim = trace.claim
    if claim.candidate_surfaces is None:
        return None
    opts = [
        i
        for i, s in enumerate(claim.candidate_surfaces)
        if s == "value" and i != claim.correct_index
    ]
    if not opts:
        return None
    return make_variant(trace, rng.choice(opts))


def corrupt_to_negation(trace: OODTrace) -> OODTrace | None:
    """Corrupt the claim to a negation-surface candidate (phantom-analog).

    Only valid when the correct answer is itself a value (so asserting a negation
    is false).
    """
    claim = trace.claim
    if claim.candidate_surfaces is None:
        return None
    if claim.candidate_surfaces[claim.correct_index] == "negation":
        return None  # correct answer is already a negation
    opts = [i for i, s in enumerate(claim.candidate_surfaces) if s == "negation"]
    if not opts:
        return None
    return make_variant(trace, opts[0])


@torch.no_grad()
def raw_claim_score(model: Any, tokenizer: Any, trace: OODTrace, device: str) -> ClaimScore:
    """Retrospective surprisal of the final claim turn (reuses Phase 0 telemetry)."""
    ids, spans = tokenize_with_provenance(tokenizer, trace.as_trace(), device)
    scores = retrospective_surprisal(model, ids, spans)
    if not scores:
        raise ValueError("no assistant spans scored")
    s = scores[-1]  # claim is the final assistant turn
    return ClaimScore(
        surface_type=trace.claim.surface_type,
        is_corrupted=trace.claim.is_corrupted,
        mean_neglogp=s.mean_neglogp,
        max_neglogp=s.max_neglogp,
        value_only_neglogp=s.slot_only_neglogp,
    )


@torch.no_grad()
def contrastive_claim_score(
    model: Any, tokenizer: Any, trace: OODTrace, device: str
) -> ClaimScore | None:
    """Candidate-renormalized score of the emitted claim (generalized Phase 0b).

    Returns None for free-text claims (no enumerable candidate set).
    """
    claim = trace.claim
    if claim.candidate_contents is None:
        return None

    prefix_text = render_chatml(trace.turns[: claim.turn_index])

    ids_list: list[torch.Tensor] = []
    cpos_list: list[list[int]] = []
    vpos_list: list[list[int]] = []
    for content, value in zip(claim.candidate_contents, claim.candidate_values):
        ids, cpos, vpos = _encode_candidate(tokenizer, prefix_text, content, value)
        ids_list.append(ids)
        cpos_list.append(cpos)
        vpos_list.append(vpos)

    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = 0
    n = len(ids_list)
    max_L = max(x.shape[1] for x in ids_list)
    batch = torch.full((n, max_L), pad, dtype=torch.long)
    mask = torch.zeros((n, max_L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        L = x.shape[1]
        batch[i, :L] = x[0]
        mask[i, :L] = 1
    batch = batch.to(device)
    mask = mask.to(device)

    logits = model(batch, attention_mask=mask).logits

    lp: dict[int, dict[str, float]] = {}
    for i in range(n):
        seq = ids_list[i][0]
        seq_logits = logits[i]
        m_mean, m_min = _mean_lp(seq_logits, seq, cpos_list[i])
        if vpos_list[i]:
            v_mean, _ = _mean_lp(seq_logits, seq, vpos_list[i])
        else:
            v_mean = m_mean
        lp[i] = {"mean": m_mean, "max": m_min, "value": v_mean}

    e = claim.emitted_index
    out: dict[str, float] = {}
    for agg in ("mean", "max", "value"):
        allv = [lp[i][agg] for i in range(n)]
        out[agg] = -(lp[e][agg] - logsumexp(allv))

    return ClaimScore(
        surface_type=claim.surface_type,
        is_corrupted=claim.is_corrupted,
        mean_neglogp=out["mean"],
        max_neglogp=out["max"],
        value_only_neglogp=out["value"],
    )
