"""Tests for Phase 2 hole-rehearsal data + collator (CPU, no peft/model)."""

from __future__ import annotations

import torch

from source_monitor.llm.train.data import HoleCollator, build_examples


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


def test_build_clean_examples():
    tok = _CharTokenizer()
    exs = build_examples(tok, seed=1, n=8, mode="clean")
    assert exs
    for e in exs:
        assert e.asst_spans and e.label_span_idxs == list(range(len(e.asst_spans)))
        for s, en in e.asst_spans:
            assert 0 <= s < en <= int(e.input_ids.shape[0])


def test_collator_no_holes_supervises_all_content():
    tok = _CharTokenizer()
    exs = build_examples(tok, seed=2, n=6, mode="clean")[:4]
    out = HoleCollator(p_hole=0.0, pad_token_id=0)(exs)
    for b, e in enumerate(exs):
        L = int(e.input_ids.shape[0])
        assert out["attention_mask"][b, :L].sum() == L  # nothing holed
        supervised = (out["labels"][b] != -100).sum().item()
        expected = sum(en - s for s, en in e.asst_spans)
        assert supervised == expected


def test_collator_holes_are_hard_and_never_supervised():
    tok = _CharTokenizer()
    exs = build_examples(tok, seed=3, n=6, mode="clean")[:4]
    out = HoleCollator(p_hole=1.0, pad_token_id=0, seed=0)(exs)
    for b, e in enumerate(exs):
        # at least one span remains supervised
        assert (out["labels"][b] != -100).any()
        # invariant: any supervised token is attendable (never both holed and supervised)
        sup = out["labels"][b] != -100
        assert torch.all(out["attention_mask"][b][sup] == 1)
        # with p_hole=1, at least one span is holed (attn 0 somewhere inside content)
        assert (out["attention_mask"][b, : int(e.input_ids.shape[0])] == 0).any()


def test_corrupt_exposure_supervises_only_post_corruption():
    tok = _CharTokenizer()
    exs = build_examples(tok, seed=5, n=60, mode="corrupt")
    assert exs, "expected some corrupted traces"
    # at least one example supervises a strict suffix of its assistant spans
    assert any(
        e.label_span_idxs and min(e.label_span_idxs) > 0
        and len(e.label_span_idxs) < len(e.asst_spans)
        for e in exs
    )


def test_build_deterministic():
    tok = _CharTokenizer()
    a = build_examples(tok, seed=9, n=6, mode="clean")
    b = build_examples(tok, seed=9, n=6, mode="clean")
    assert [x.asst_spans for x in a] == [x.asst_spans for x in b]
