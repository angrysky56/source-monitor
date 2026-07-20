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

    # Flagging. A purely RELATIVE z-rule fires on ~96% of clean traces (with ~5
    # spans the max is almost always >=1.5 sigma), so the default is an ABSOLUTE
    # floor calibrated on held-out clean traces — that directly controls the idle
    # false-excision rate. "both" ANDs the two; "zscore" is the old behaviour.
    flag_mode: str = "absolute"  # "absolute" | "zscore" | "both"
    k_threshold: float = 1.5
    calib_quantile: float = 0.99  # per-span quantile of clean scores
    calib_n: int = 40
    calib_seed: int = 7  # held out from the eval seeds

    max_new_tokens: int = 48  # headroom even with thinking disabled
    conditions: tuple[str, ...] = ("monitor_off", "monitor_on", "oracle_excise")
    results_dir: str = "results"
