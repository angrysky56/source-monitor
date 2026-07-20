"""Phase 2 evaluation: the bsi analog, competence tax, and detection-survives.

Metrics (generation-free, via candidate ranking / teacher-forced surprisal):
- competence      : final-answer accuracy on CLEAN traces (argmax candidate == correct)
- planted_acc     : final-answer accuracy when a false self-emission is in context
- bsi             : competence - planted_acc  (behavioral cost of a self-lie)
- detect_auroc    : value-only surprisal AUROC separating a corrupted FINAL claim
                    from the genuine one (Phase 0/1 detection, must survive LoRA)

Run per model (base weights vs a LoRA adapter) and compare arms.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from source_monitor.llm.ood import entity_prose
from source_monitor.llm.ood.base import corrupt_to_value, raw_claim_score
from source_monitor.llm.telemetry import _encode_candidate, _mean_lp, render_chatml
from source_monitor.metrics import auroc


@torch.no_grad()
def _ranks_correct(model: Any, tok: Any, trace, device: str) -> bool:
    """Does the model rank the CORRECT final-claim candidate highest (mean logp)?"""
    claim = trace.claim
    prefix = render_chatml(trace.turns[: claim.turn_index])
    best_i, best_lp = -1, -1e30
    for i, (content, value) in enumerate(zip(claim.candidate_contents, claim.candidate_values)):
        ids, cpos, _v = _encode_candidate(tok, prefix, content, value)
        logits = model(ids.to(device)).logits[0]
        mean_lp, _ = _mean_lp(logits, ids[0], cpos)
        if mean_lp > best_lp:
            best_lp, best_i = mean_lp, i
    return best_i == claim.correct_index


@torch.no_grad()
def rank_accuracy(model: Any, tok: Any, traces, device: str) -> float:
    if not traces:
        return float("nan")
    return sum(_ranks_correct(model, tok, tr, device) for tr in traces) / len(traces)


@torch.no_grad()
def _ranks_correct_excised(model: Any, tok: Any, trace, device: str) -> bool | None:
    """Like _ranks_correct, but the planted lie's tokens are attention-masked out
    (excised → turned into a hole) before scoring. Tests whether the model
    re-derives the correct answer once the lie is removed from attention."""
    from source_monitor.llm.provenance import tokenize_with_provenance
    from source_monitor.llm.task_render import Trace

    claim = trace.claim
    ci = trace.meta.get("corrupt_turn_index")
    if ci is None:
        return None
    prefix_turns = trace.turns[: claim.turn_index]
    pre = Trace(turns=prefix_turns, query_object="", ground_truth_final="", op_kinds=[], task=None)
    _ids, spans = tokenize_with_provenance(tok, pre, device)
    if ci >= len(spans):
        return None
    hs, he = spans[ci].start_token, spans[ci].end_token  # lie's content token span
    prefix_text = render_chatml(prefix_turns)

    best_i, best_lp = -1, -1e30
    for i, (content, value) in enumerate(zip(claim.candidate_contents, claim.candidate_values)):
        ids, cpos, _v = _encode_candidate(tok, prefix_text, content, value)
        attn = torch.ones_like(ids)
        attn[0, hs:he] = 0  # excise the lie
        logits = model(ids.to(device), attention_mask=attn.to(device)).logits[0]
        mean_lp, _ = _mean_lp(logits, ids[0], cpos)
        if mean_lp > best_lp:
            best_lp, best_i = mean_lp, i
    return best_i == claim.correct_index


@torch.no_grad()
def rank_accuracy_excised(model: Any, tok: Any, traces, device: str) -> float:
    res = [_ranks_correct_excised(model, tok, tr, device) for tr in traces]
    res = [r for r in res if r is not None]
    return sum(res) / len(res) if res else float("nan")


@torch.no_grad()
def evaluate(model: Any, tok: Any, seed: int, n: int, device: str) -> dict[str, float]:
    clean = entity_prose.generate(seed, n)
    planted = entity_prose.generate(seed, n, corrupt_mid=True)  # lie in a mid ack
    planted = [t for t in planted if t.meta.get("corrupt_turn_index") is not None]

    competence = rank_accuracy(model, tok, clean, device)
    planted_acc = rank_accuracy(model, tok, planted, device)

    # detection: corrupt the FINAL claim, value-only surprisal AUROC vs genuine
    rng = random.Random(seed + 3)
    gen = [raw_claim_score(model, tok, tr, device).value_only_neglogp for tr in clean]
    cor = []
    for tr in clean:
        cv = corrupt_to_value(tr, rng)
        if cv is not None:
            cor.append(raw_claim_score(model, tok, cv, device).value_only_neglogp)
    detect = auroc(cor + gen, [1] * len(cor) + [0] * len(gen)) if cor else float("nan")

    return {
        "competence": competence,
        "planted_acc": planted_acc,
        "bsi": competence - planted_acc,
        "detect_auroc": detect,
        "n_clean": len(clean), "n_planted": len(planted),
    }


def load_for_eval(model_name: str, device: str, dtype: str, adapter_dir: str | None):
    """Load base weights, optionally applying a LoRA adapter."""
    from source_monitor.llm.cache import load_model

    model, tok, meta = load_model(model_name, device=device, dtype=dtype, enable_thinking=False)
    if adapter_dir:
        from peft import PeftModel  # lazy: only needed when evaluating an adapter

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok, meta
