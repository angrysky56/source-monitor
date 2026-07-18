"""Typed configuration for Phase 0 (and later phase) LLM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskConfig:
    """Entity-tracking task parameters (mirrors source_monitor.task.generate_task kwargs)."""
    n_ops: int = 8
    n_objects: int = 4
    n_containers: int = 3
    remove_prob: float = 0.25
    label: str = "primary"  # "primary" or "hard"


PRIMARY_TASK = TaskConfig(n_ops=8, n_objects=4, n_containers=3, label="primary")
HARD_TASK = TaskConfig(n_ops=12, n_objects=6, n_containers=4, label="hard")

# Aggregation is not a config choice — all three are always computed (A4).
AGGREGATION_METHODS = ("mean", "max", "slot_only")


@dataclass(frozen=True)
class Phase0Config:
    """Full Phase 0 experiment specification."""
    model_names: tuple[str, ...] = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
    device: str = "cuda"
    dtype: str = "bfloat16"
    task_configs: tuple[TaskConfig, ...] = (PRIMARY_TASK, HARD_TASK)
    n_traces: int = 400
    seeds: tuple[int, ...] = (42, 137, 2024)
    corruption_types: tuple[str, ...] = ("ghost", "mislocation", "phantom")
    results_dir: str = "results"
    enable_thinking: bool = False  # A1: pinned off for Qwen3
