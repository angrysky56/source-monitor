"""Layer 1: Telemetry / detection signal.

Computes the retrospective surprisal of self-spans (assistant content) using
the token log probabilities from teacher-forced forward passes.

A4: Computes mean, max, and location-slot-only aggregations side by side.

Phase 0b: adds candidate-renormalized (likelihood-ratio) slot scoring, which
cancels the absence-assertion prior bias by comparing the emitted claim against
the trace's own candidate set at each slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from source_monitor.llm.provenance import SpanAnnotation

if TYPE_CHECKING:
    from source_monitor.llm.task_render import Trace


@dataclass
class SpanScore:
    """The computed surprisal scores for a single assistant turn."""

    step_index: int
    is_corrupted: bool
    claim_surface: str
    mean_neglogp: float  # (a) mean over all content tokens
    max_neglogp: float  # (b) max over all content tokens
    slot_only_neglogp: float  # (c) mean over only location-slot tokens


@torch.no_grad()
def retrospective_surprisal(
    model: Any,
    input_ids: Tensor,
    spans: list[SpanAnnotation],
) -> list[SpanScore]:
    """Perform a teacher-forced forward pass and compute span-level surprisals.

    Returns one SpanScore per assistant turn.
    """
    device = input_ids.device

    # Forward pass: (1, L, V)
    outputs = model(input_ids)
    logits = outputs.logits

    # Calculate log probabilities of vocabulary at each position
    # Logprob of token at index pos is log_probs[0, pos - 1, input_ids[0, pos]]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    scores: list[SpanScore] = []

    for span in spans:
        if span.kind != "assistant":
            continue

        # Extract content tokens: [start_token, end_token)
        content_tokens = list(range(span.start_token, span.end_token))
        if not content_tokens:
            continue

        # Get negative logprob for each content token
        neglogps = []
        for pos in content_tokens:
            if pos <= 0:  # cannot predict the first token
                continue
            token_id = int(input_ids[0, pos])
            lp = float(log_probs[0, pos - 1, token_id])
            neglogps.append(-lp)

        if not neglogps:
            continue

        # Aggregation (a): Mean over content tokens
        mean_val = sum(neglogps) / len(neglogps)

        # Aggregation (b): Max token neglogp (min logp)
        max_val = max(neglogps)

        # Aggregation (c): Location-slot only
        slot_neglogps = []
        if (
            span.location_start_token is not None
            and span.location_end_token is not None
        ):
            slot_tokens = list(
                range(span.location_start_token, span.location_end_token)
            )
            for pos in slot_tokens:
                if pos <= 0:
                    continue
                token_id = int(input_ids[0, pos])
                lp = float(log_probs[0, pos - 1, token_id])
                slot_neglogps.append(-lp)

        if slot_neglogps:
            slot_val = sum(slot_neglogps) / len(slot_neglogps)
        else:
            # Fallback to mean content score if slot is empty or not annotated
            slot_val = mean_val

        scores.append(
            SpanScore(
                step_index=span.step_index if span.step_index is not None else -1,
                is_corrupted=span.is_corrupted,
                claim_surface=(
                    span.claim_surface if span.claim_surface is not None else "unknown"
                ),
                mean_neglogp=mean_val,
                max_neglogp=max_val,
                slot_only_neglogp=slot_val,
            )
        )

    return scores


def logsumexp(vals: list[float]) -> float:
    """Numerically stable logsumexp in pure Python."""
    max_val = max(vals)
    return max_val + math.log(sum(math.exp(v - max_val) for v in vals))


# --- Phase 0b: contrastive slot scoring ------------------------------------
#
# ChatML rendering is done in Python, byte-identical to the CHATML_TEMPLATE
# override installed by cache.load_model. The Jinja `apply_chat_template` call
# is O(turns) per invocation; the previous implementation re-ran it once per
# candidate per turn (O(turns^2 * |C|) Jinja renders per trace), which is what
# made the contrastive sweep appear to hang. Building the string directly
# removes Jinja from the hot loop entirely; `test_telemetry_llm.py` asserts the
# two renders are byte-equal against the real tokenizer.

_CHATML_MESSAGE = "<|im_start|>{role}\n{content}<|im_end|>\n"
_ASSISTANT_HEADER = "<|im_start|>assistant\n"
_ASSISTANT_FOOTER = "<|im_end|>\n"


def render_chatml(turns: Any) -> str:
    """Render a list of turns to the ChatML string (matches cache.CHATML_TEMPLATE).

    Each turn must expose `.role` and `.content`. Equivalent to
    tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)
    when the ChatML override template is installed.
    """
    return "".join(
        _CHATML_MESSAGE.format(role=t.role, content=t.content) for t in turns
    )


def _encode_candidate(
    tokenizer: Any,
    prefix_text: str,
    cand_content: str,
    location_text: str,
) -> tuple[Tensor, list[int], list[int]]:
    """Tokenize `prefix_text` + one assistant candidate turn.

    Returns (input_ids [1, L], content token positions, slot token positions).
    Positions are absolute indices into the returned sequence. The candidate
    content is placed deterministically right after the assistant header, so its
    character span needs no searching.
    """
    full_text = prefix_text + _ASSISTANT_HEADER + cand_content + _ASSISTANT_FOOTER
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()

    c_start = len(prefix_text) + len(_ASSISTANT_HEADER)
    c_end = c_start + len(cand_content)

    content_pos = [
        i
        for i, (a, b) in enumerate(offsets)
        if not (a == b == 0) and a >= c_start and b <= c_end
    ]

    slot_pos: list[int] = []
    loc_idx = cand_content.find(location_text)
    if loc_idx != -1:
        l_start = c_start + loc_idx
        l_end = l_start + len(location_text)
        slot_pos = [
            i
            for i in content_pos
            if offsets[i][0] >= l_start and offsets[i][1] <= l_end
        ]

    return input_ids, content_pos, slot_pos


def _mean_lp(logits_seq: Tensor, ids_seq: Tensor, positions: list[int]) -> tuple[float, float]:
    """Mean and min log-prob of tokens at `positions` given preceding context.

    logits_seq: (L, V) for one sequence. ids_seq: (L,) token ids for the same
    sequence. Returns (mean_lp, min_lp); min_lp is the most surprising token.
    """
    lps: list[float] = []
    for pos in positions:
        if pos <= 0:
            continue
        row = logits_seq[pos - 1].float()
        lp = float(F.log_softmax(row, dim=-1)[int(ids_seq[pos])])
        lps.append(lp)
    if not lps:
        return 0.0, 0.0
    return sum(lps) / len(lps), min(lps)


@torch.no_grad()
def contrastive_slot_scores(
    model: Any,
    tokenizer: Any,
    trace: "Trace",
    device: str,
) -> list[SpanScore]:
    """Compute candidate-renormalized surprisal at each assistant slot (Phase 0b).

    For emission slot s the candidate set is C(s) = {containers named in this
    trace} u {"nowhere"} (per implementation_plan Phase 0b; NOT the global
    8-box pool, which dilutes the renormalization with never-seen boxes). The
    score is -log of the renormalized probability of the emitted claim within
    C(s), which cancels surface-form priors shared by all candidates.
    """
    from source_monitor.llm.task_render import CONTAINER_NAMES

    # C(s): containers actually named anywhere in this trace, plus "nowhere".
    trace_containers = [
        c for c in CONTAINER_NAMES if any(c in t.content for t in trace.turns)
    ]
    candidates = trace_containers + ["nowhere"]

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    scores: list[SpanScore] = []

    for turn_idx, turn in enumerate(trace.turns):
        if not turn.is_self or turn.role != "assistant":
            continue

        step_k = turn.step_index
        assert step_k is not None

        # Shared prefix rendered once (Python, no Jinja) for all candidates.
        prefix_text = render_chatml(trace.turns[:turn_idx])

        cand_ids_list: list[Tensor] = []
        cand_content_pos: list[list[int]] = []
        cand_slot_pos: list[list[int]] = []

        for c in candidates:
            if c == "nowhere":
                cand_content = f"The {trace.query_object} is nowhere."
            else:
                cand_content = f"The {trace.query_object} is in {c}."

            ids, content_pos, slot_pos = _encode_candidate(
                tokenizer, prefix_text, cand_content, c
            )
            cand_ids_list.append(ids)
            cand_content_pos.append(content_pos)
            cand_slot_pos.append(slot_pos)

        # Batched, right-padded forward pass over the candidate variants.
        max_L = max(ids.shape[1] for ids in cand_ids_list)
        batch_ids = torch.full(
            (len(candidates), max_L), pad_token_id, dtype=torch.long
        )
        batch_mask = torch.zeros((len(candidates), max_L), dtype=torch.long)
        for i, ids in enumerate(cand_ids_list):
            L_i = ids.shape[1]
            batch_ids[i, :L_i] = ids[0]
            batch_mask[i, :L_i] = 1

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)

        outputs = model(batch_ids, attention_mask=batch_mask)
        logits = outputs.logits  # (|C|, max_L, V)

        lp_candidates: dict[str, dict[str, float]] = {}
        for i, c in enumerate(candidates):
            ids_seq = cand_ids_list[i][0]
            logits_seq = logits[i]

            mean_lp, min_lp = _mean_lp(logits_seq, ids_seq, cand_content_pos[i])
            if cand_slot_pos[i]:
                slot_lp, _ = _mean_lp(logits_seq, ids_seq, cand_slot_pos[i])
            else:
                slot_lp = mean_lp

            lp_candidates[c] = {"mean": mean_lp, "max": min_lp, "slot_only": slot_lp}

        # Free the large logits tensor before the next turn's forward pass.
        del outputs, logits, batch_ids, batch_mask

        emitted_loc = turn.location_text
        assert (
            emitted_loc in lp_candidates
        ), f"Emitted location {emitted_loc!r} not in candidates {candidates}"

        agg_scores: dict[str, float] = {}
        for agg in ("mean", "max", "slot_only"):
            lp_emitted = lp_candidates[emitted_loc][agg]
            lp_all = [lp_candidates[c][agg] for c in candidates]
            agg_scores[agg] = -(lp_emitted - logsumexp(lp_all))

        scores.append(
            SpanScore(
                step_index=step_k,
                is_corrupted=turn.is_corrupted,
                claim_surface=(
                    turn.claim_surface if turn.claim_surface is not None else "unknown"
                ),
                mean_neglogp=agg_scores["mean"],
                max_neglogp=agg_scores["max"],
                slot_only_neglogp=agg_scores["slot_only"],
            )
        )

    return scores
