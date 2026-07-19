"""Confirmatory variant of factual_qa where the answer is stated in-context.

Thin wrapper so the runner can treat it as its own domain. If the generative
signal detects errors here (high AUROC) but not on plain factual_qa (~chance),
the Phase 1 boundary is derivability, not the factual domain.
"""

from __future__ import annotations

from source_monitor.llm.ood import factual_qa
from source_monitor.llm.ood.base import OODTrace


def generate(seed: int, n: int, **kwargs) -> list[OODTrace]:
    return factual_qa.generate(seed, n, grounded=True, **kwargs)
