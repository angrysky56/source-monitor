"""Tests for Phase 1 OOD generators and the generalized claim scorer."""

from __future__ import annotations

import random

import torch

from source_monitor.llm.ood import arithmetic, code_trace, entity_prose, factual_qa
from source_monitor.llm.ood.base import (
    contrastive_claim_score,
    corrupt_to_negation,
    corrupt_to_value,
    raw_claim_score,
)

ENUM = {"entity_prose": entity_prose, "arithmetic": arithmetic, "factual_qa": factual_qa}


class _CharTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chat_turns, tokenize=False, add_generation_prompt=False):
        return "".join(
            f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in chat_turns
        )

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        n = len(text)
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "offset_mapping": torch.tensor([[i, i + 1] for i in range(n)]).unsqueeze(0),
        }


class _MockOut:
    def __init__(self, logits):
        self.logits = logits


class _MockModel:
    def __call__(self, input_ids, *a, **k):
        b, L = input_ids.shape
        return _MockOut(torch.zeros((b, L, 2048), dtype=torch.float32))


def test_all_generators_deterministic():
    for mod in (entity_prose, arithmetic, factual_qa, code_trace):
        a = mod.generate(7, 20)
        b = mod.generate(7, 20)
        assert [[t.content for t in tr.turns] for tr in a] == \
               [[t.content for t in tr.turns] for tr in b]


def test_generators_produce_both_surfaces():
    for name, mod in ENUM.items():
        traces = mod.generate(1, 120)
        surfaces = {tr.claim.surface_type for tr in traces}
        assert "value" in surfaces, f"{name}: no value-correct traces"
        assert "negation" in surfaces, f"{name}: no negation-correct traces"


def test_claim_is_final_assistant_turn():
    for mod in (entity_prose, arithmetic, factual_qa, code_trace):
        for tr in mod.generate(3, 10):
            assert tr.claim.turn_index == len(tr.turns) - 1
            last = tr.turns[-1]
            assert last.role == "assistant" and last.is_self


def test_enumerable_corruptions_only_change_claim_turn():
    for name, mod in ENUM.items():
        rng = random.Random(5)
        for clean in mod.generate(5, 30):
            for variant in (corrupt_to_value(clean, rng), corrupt_to_negation(clean)):
                if variant is None:
                    continue
                assert variant.claim.is_corrupted
                for i, (c, v) in enumerate(zip(clean.turns, variant.turns)):
                    if i != clean.claim.turn_index:
                        assert c.content == v.content, f"{name}: non-claim turn changed"


def test_entity_prose_candidates_are_in_context():
    """Anti-confound: every value candidate location appears in the trace text."""
    for tr in entity_prose.generate(2, 40):
        ctx = " ".join(t.content for t in tr.turns[:-1])
        for val, surf in zip(tr.claim.candidate_values, tr.claim.candidate_surfaces):
            if surf == "value":
                assert val in ctx, f"out-of-context candidate {val!r}"


def test_arithmetic_running_total_is_correct():
    # The reported running totals must match a Python recomputation.
    for tr in arithmetic.generate(9, 40):
        total = None
        for t in tr.turns:
            if t.role == "user" and t.content.startswith("Start with"):
                total = int(t.content.split()[-1].rstrip("."))
            elif t.role == "user" and t.content.startswith("Add"):
                total += int(t.content.split()[-1].rstrip("."))
            elif t.role == "user" and t.content.startswith("Subtract"):
                total -= int(t.content.split()[-1].rstrip("."))
            elif t.role == "user" and t.content.startswith("Multiply"):
                total *= int(t.content.split()[-1].rstrip("."))
        # for value-correct traces the correct candidate equals the final total
        if not tr.meta["negation_correct"]:
            assert tr.claim.candidate_values[tr.claim.correct_index] == str(total)


def test_code_ground_truth_and_corruptions():
    rng = random.Random(0)
    for tr in code_trace.generate(11, 40):
        val = tr.meta["value"]
        if val is not None:
            assert f"is {val}." in tr.turns[-1].content
        cv = code_trace.corrupt_value(tr, rng)
        assert cv.claim.is_corrupted and cv.turns[-1].content != tr.turns[-1].content
        cn = code_trace.corrupt_negation(tr)
        if tr.meta["value"] is None:
            assert cn is None  # can't claim an undefined var is undefined as a lie
        else:
            assert cn is not None and cn.claim.surface_type == "negation"


def test_scorers_run_on_mock():
    tok, model = _CharTokenizer(), _MockModel()
    # enumerable domain: both scorers return finite scores
    tr = entity_prose.generate(1, 1)[0]
    rs = raw_claim_score(model, tok, tr, "cpu")
    cs = contrastive_claim_score(model, tok, tr, "cpu")
    assert rs is not None and cs is not None
    for s in (rs, cs):
        for a in (s.mean_neglogp, s.max_neglogp, s.value_only_neglogp):
            assert a == a  # not NaN
    # free-text domain: contrastive is skipped
    ct = code_trace.generate(1, 1)[0]
    assert contrastive_claim_score(model, tok, ct, "cpu") is None
    assert raw_claim_score(model, tok, ct, "cpu") is not None
