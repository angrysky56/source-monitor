"""Training data + the hole-masking collator for Phase 2.

Examples are built from CLEAN (or corruption-exposure) multi-turn traces; loss is
supervised on assistant-emission CONTENT tokens only. The HoleCollator randomly
attention-masks a fraction of assistant emission CONTENT spans (hard holes) and
excludes those from the loss, so the remaining supervised emissions must be
predicted without attending to the holed ones — forcing re-derivation.

Holes touch CONTENT tokens only; the "<|im_start|>assistant\\n" header before each
emission stays visible, so the token that predicts a supervised span's first token
is never masked.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from source_monitor.llm.ood import arithmetic, entity_prose
from source_monitor.llm.provenance import tokenize_with_provenance

_DOMAIN = {"entity_prose": entity_prose, "arithmetic": arithmetic}


@dataclass
class TrainExample:
    input_ids: torch.Tensor  # (L,)
    asst_spans: list[tuple[int, int]]  # assistant emission CONTENT token ranges
    label_span_idxs: list[int]  # indices of asst_spans eligible for supervision


def build_examples(
    tokenizer: Any,
    seed: int,
    n: int,
    domains: tuple[str, ...] = ("entity_prose",),
    mode: str = "clean",  # "clean" | "corrupt"
    device: str = "cpu",
) -> list[TrainExample]:
    exs: list[TrainExample] = []
    per = max(1, n // len(domains))
    for dom in domains:
        mod = _DOMAIN[dom]
        if mode == "corrupt" and dom == "entity_prose":
            traces = mod.generate(seed, per, corrupt_mid=True)
        else:
            traces = mod.generate(seed, per)
        for tr in traces:
            ids, spans = tokenize_with_provenance(tokenizer, tr.as_trace(), device)
            asst_turn_idx = [
                i for i, s in enumerate(spans)
                if s.kind == "assistant" and s.end_token > s.start_token
            ]
            asst = [(spans[i].start_token, spans[i].end_token) for i in asst_turn_idx]
            if not asst:
                continue
            if mode == "corrupt":
                ci = tr.meta.get("corrupt_turn_index")
                if ci is None:
                    continue  # no corruption planted for this trace
                label_idxs = [j for j, ti in enumerate(asst_turn_idx) if ti > ci]
                if not label_idxs:
                    continue
            else:
                label_idxs = list(range(len(asst)))
            exs.append(TrainExample(ids[0].to("cpu"), asst, label_idxs))
    return exs


class HoleCollator:
    """Pads a batch and applies hard holes (attention_mask=0) to a random subset
    of assistant emission spans, supervising the non-holed eligible spans."""

    def __init__(self, p_hole: float, pad_token_id: int, seed: int = 0):
        self.p_hole = p_hole
        self.pad = pad_token_id if pad_token_id is not None else 0
        self.rng = random.Random(seed)

    def __call__(self, batch: list[TrainExample]) -> dict[str, torch.Tensor]:
        max_L = max(int(e.input_ids.shape[0]) for e in batch)
        B = len(batch)
        input_ids = torch.full((B, max_L), self.pad, dtype=torch.long)
        attn = torch.zeros((B, max_L), dtype=torch.long)
        labels = torch.full((B, max_L), -100, dtype=torch.long)

        for b, e in enumerate(batch):
            L = int(e.input_ids.shape[0])
            input_ids[b, :L] = e.input_ids
            attn[b, :L] = 1

            holed: set[int] = set()
            if self.p_hole > 0:
                for si in range(len(e.asst_spans)):
                    if self.rng.random() < self.p_hole:
                        holed.add(si)

            supervisable = [si for si in e.label_span_idxs if si not in holed]
            if not supervisable and e.label_span_idxs:
                keep = self.rng.choice(e.label_span_idxs)
                holed.discard(keep)
                supervisable = [keep]

            for si in holed:
                s, en = e.asst_spans[si]
                attn[b, s:en] = 0  # hard hole: content not attendable
            for si in supervisable:
                s, en = e.asst_spans[si]
                labels[b, s:en] = e.input_ids[s:en]

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def iterate_batches(examples: list[TrainExample], batch_size: int, steps: int, seed: int):
    """Yield `steps` random mini-batches (with replacement across epochs)."""
    rng = random.Random(seed)
    order: list[TrainExample] = []
    for _ in range(steps):
        if len(order) < batch_size:
            order = examples[:]
            rng.shuffle(order)
        yield order[:batch_size]
        order = order[batch_size:]
