"""Typed configuration for Phase 2 hole-rehearsal LoRA."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Qwen3 attention + MLP projections.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )


@dataclass(frozen=True)
class Phase2Config:
    model_name: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # data
    domains: tuple[str, ...] = ("entity_prose",)
    n_train: int = 400
    n_eval: int = 200

    # arms: base (no holes), drop (hole rehearsal), corrupt (corruption exposure)
    arms: tuple[str, ...] = ("base", "drop", "corrupt")
    p_hole: float = 0.5  # per-earlier-emission hole probability for the drop arm

    # optimization
    seeds: tuple[int, ...] = (42, 137, 2024)
    steps: int = 300
    lr: float = 1e-4
    batch_size: int = 2  # 12GB 3060: keep the per-step activation/logits footprint small
    grad_accum: int = 1
    warmup: int = 20

    results_dir: str = "results"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
