"""Layer 1: Telemetry / detection signal.

Computes the retrospective surprisal of self-spans (assistant content) using
the token log probabilities from teacher-forced forward passes.

A4: Computes mean, max, and location-slot-only aggregations side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from source_monitor.llm.provenance import SpanAnnotation


@dataclass
class SpanScore:
    """The computed surprisal scores for a single assistant turn."""
    step_index: int
    is_corrupted: bool
    claim_surface: str
    mean_neglogp: float          # (a) mean over all content tokens
    max_neglogp: float           # (b) max over all content tokens
    slot_only_neglogp: float     # (c) mean over only location-slot tokens


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
            slot_tokens = list(range(span.location_start_token, span.location_end_token))
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
                claim_surface=span.claim_surface if span.claim_surface is not None else "unknown",
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


@torch.no_grad()
def contrastive_slot_scores(
    model: Any,
    tokenizer: Any,
    trace: Trace,
    device: str,
) -> list[SpanScore]:
    """Compute candidate-renormalized surprisal at each assistant turn slot (Phase 0b).

    C(s) = CONTAINER_NAMES ∪ ["nowhere"].
    Optimized: Evaluates all candidates in a single batched forward pass.
    """
    import math
    from source_monitor.llm.task_render import CONTAINER_NAMES, Turn, Trace
    from source_monitor.llm.provenance import tokenize_with_provenance
    
    candidates = list(CONTAINER_NAMES) + ["nowhere"]
    scores: list[SpanScore] = []
    
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    
    # Iterate over all assistant turns
    for turn_idx, turn in enumerate(trace.turns):
        if not turn.is_self or turn.role != "assistant":
            continue
            
        step_k = turn.step_index
        assert step_k is not None
        
        # Prepare list to batch tokenize
        cand_ids_list = []
        cand_spans_list = []
        
        for c in candidates:
            # Construct candidate content
            if c == "nowhere":
                cand_content = f"The {trace.query_object} is nowhere."
                claim_surf = "nowhere"
            else:
                cand_content = f"The {trace.query_object} is in {c}."
                claim_surf = "container"
                
            # Create candidate trace prefix + candidate turn
            cand_turns = []
            for t in trace.turns[:turn_idx]:
                cand_turns.append(
                    Turn(
                        role=t.role,
                        content=t.content,
                        is_self=t.is_self,
                        step_index=t.step_index,
                        is_corrupted=t.is_corrupted,
                        claim_surface=t.claim_surface,
                        location_text=t.location_text,
                    )
                )
            cand_turns.append(
                Turn(
                    role="assistant",
                    content=cand_content,
                    is_self=True,
                    step_index=step_k,
                    claim_surface=claim_surf,
                    location_text=c,
                )
            )
            
            cand_trace = Trace(
                turns=cand_turns,
                query_object=trace.query_object,
                ground_truth_final=trace.ground_truth_final,
                op_kinds=trace.op_kinds,
                task=trace.task,
            )
            
            # Tokenize on CPU to allow padding
            cand_ids, cand_spans = tokenize_with_provenance(tokenizer, cand_trace, "cpu", skip_prefix_check=True)
            cand_ids_list.append(cand_ids)
            cand_spans_list.append(cand_spans)
            
        # Batched padding and collation
        max_L = max(ids.shape[1] for ids in cand_ids_list)
        batch_ids = torch.full((len(candidates), max_L), pad_token_id, dtype=torch.long)
        batch_mask = torch.zeros((len(candidates), max_L), dtype=torch.long)
        
        for i, cand_ids in enumerate(cand_ids_list):
            L_i = cand_ids.shape[1]
            batch_ids[i, :L_i] = cand_ids[0]
            batch_mask[i, :L_i] = 1
            
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        
        # Batched forward pass: (9, max_L, V)
        outputs = model(batch_ids, attention_mask=batch_mask)
        
        lp_candidates = {}
        
        for i, c in enumerate(candidates):
            span = cand_spans_list[i][-1]
            cand_ids = cand_ids_list[i]
            
            # Content tokens range: [start_token, end_token)
            content_tokens = list(range(span.start_token, span.end_token))
            neg_logps_content = []
            
            if content_tokens:
                # Slice logits for content tokens: shape (len(content_tokens), V)
                content_logits = outputs.logits[i, [pos - 1 for pos in content_tokens], :].float()
                content_log_probs = F.log_softmax(content_logits, dim=-1)
                
                for idx, pos in enumerate(content_tokens):
                    if pos <= 0:
                        continue
                    token_id = int(cand_ids[0, pos])
                    lp = float(content_log_probs[idx, token_id])
                    neg_logps_content.append(lp)
                
            mean_lp = sum(neg_logps_content) / len(neg_logps_content) if neg_logps_content else 0.0
            max_neg_lp = min(neg_logps_content) if neg_logps_content else 0.0
            
            # Slot tokens range: [location_start_token, location_end_token)
            slot_tokens = []
            if span.location_start_token is not None and span.location_end_token is not None:
                slot_tokens = list(range(span.location_start_token, span.location_end_token))
                
            slot_logps = []
            if slot_tokens:
                # Slice logits for slot tokens: shape (len(slot_tokens), V)
                slot_logits = outputs.logits[i, [pos - 1 for pos in slot_tokens], :].float()
                slot_log_probs = F.log_softmax(slot_logits, dim=-1)
                
                for idx, pos in enumerate(slot_tokens):
                    if pos <= 0:
                        continue
                    token_id = int(cand_ids[0, pos])
                    lp = float(slot_log_probs[idx, token_id])
                    slot_logps.append(lp)
                
            slot_lp = sum(slot_logps) / len(slot_logps) if slot_logps else mean_lp
            
            lp_candidates[c] = {
                "mean": mean_lp,
                "max": max_neg_lp,
                "slot_only": slot_lp,
            }
            
        emitted_loc = turn.location_text
        assert emitted_loc in lp_candidates, f"Emitted location {emitted_loc!r} not in candidates"
        
        agg_scores = {}
        for agg in ("mean", "max", "slot_only"):
            lp_emitted = lp_candidates[emitted_loc][agg]
            lp_all = [lp_candidates[c][agg] for c in candidates]
            
            lse = logsumexp(lp_all)
            agg_scores[agg] = - (lp_emitted - lse)
            
        scores.append(
            SpanScore(
                step_index=step_k,
                is_corrupted=turn.is_corrupted,
                claim_surface=turn.claim_surface if turn.claim_surface is not None else "unknown",
                mean_neglogp=agg_scores["mean"],
                max_neglogp=agg_scores["max"],
                slot_only_neglogp=agg_scores["slot_only"],
            )
        )
        
    return scores

