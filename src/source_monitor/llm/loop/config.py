"""Typed configuration for the Phase 3 closed loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase3Config:
    model_name: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda"
    dtype: str = "bfloat16"

    n_traces: int = 100
    seeds: tuple[int, ...] = (42, 137, 2024)

    # Flagging: within-trace z-score over self-span scores. Self-contained, no
    # external calibration set; k trades false excisions against misses.
    k_threshold: float = 1.5

    max_new_tokens: int = 48  # headroom even with thinking disabled
    conditions: tuple[str, ...] = ("monitor_off", "monitor_on", "oracle_excise")
    results_dir: str = "results"
