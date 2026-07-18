"""Phase 0 experiment orchestrator.

Runs the zero-shot retrospective surprisal detection experiment over Qwen3 models.
Saves results to results/llm_phase0_results.jsonl.

A2: Computes matched-surface stratified AUROCs.
A3: Computes paired delta scoring and separates the three genuine populations.
A4: Reports all three aggregations (mean, max, slot-only) side by side.
A5: Records environment and model provenance metadata.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from source_monitor.metrics import auroc
from source_monitor.task import generate_dataset
from source_monitor.llm.config import TaskConfig, Phase0Config, PRIMARY_TASK, HARD_TASK
from source_monitor.llm.cache import load_model, ModelMeta
from source_monitor.llm.task_render import render_trace
from source_monitor.llm.corruption import inject_all_types, CorruptionRecord
from source_monitor.llm.provenance import tokenize_with_provenance
from source_monitor.llm.telemetry import retrospective_surprisal, SpanScore


@dataclass
class AggregationResult:
    """Surprisal detection metrics for a single aggregation method (A4)."""
    # Pooled AUROCs
    pooled_auroc: dict[str, float] = field(default_factory=dict)
    # Matched-surface stratified AUROCs (A2)
    matched_surface_auroc: dict[str, float] = field(default_factory=dict)
    # Paired deltas (A3)
    mean_paired_delta: dict[str, float] = field(default_factory=dict)
    # Population means (A3)
    mean_genuine_clean: float = 0.0
    mean_genuine_before: dict[str, float] = field(default_factory=dict)
    mean_genuine_cascade: dict[str, float] = field(default_factory=dict)
    mean_corrupt: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResultRecord:
    """Full flat result record saved to JSONL (A5)."""
    model_name: str
    task_label: str  # "primary" or "hard"
    seed: int
    n_traces: int
    mean: AggregationResult
    max: AggregationResult
    slot_only: AggregationResult
    meta: dict[str, Any]
    wall_seconds: float


def run_model_experiment(
    model: Any,
    tokenizer: Any,
    meta: ModelMeta,
    task_cfg: TaskConfig,
    seed: int,
    n_traces: int,
    corruption_types: tuple[str, ...],
    device: str,
) -> ExperimentResultRecord:
    """Run Phase 0 analysis for a single model, task complexity, and seed."""
    start_time = time.time()
    
    # 1. Generate tasks deterministically
    tasks = generate_dataset(
        n_tasks=n_traces,
        base_seed=seed,
        n_ops=task_cfg.n_ops,
        n_objects=task_cfg.n_objects,
        n_containers=task_cfg.n_containers,
        remove_prob=task_cfg.remove_prob,
    )
    
    # 2. Render to trace sets
    trace_rng = random.Random(seed + 1000)
    
    # Storage for all populations (A3)
    # Lists of scores grouped by aggregation method ("mean", "max", "slot_only")
    clean_genuine_scores = defaultdict(list)
    clean_genuine_surfaces = []
    
    # Corrupt scores grouped by: agg -> corr_type -> list of scores
    corrupt_scores = {agg: {t: [] for t in corruption_types} for agg in ("mean", "max", "slot_only")}
    # Before corruption scores: agg -> corr_type -> list of scores
    before_scores = {agg: {t: [] for t in corruption_types} for agg in ("mean", "max", "slot_only")}
    # Cascade (after corruption) scores: agg -> corr_type -> list of scores
    cascade_scores = {agg: {t: [] for t in corruption_types} for agg in ("mean", "max", "slot_only")}
    # Paired deltas (corr - clean_twin): agg -> corr_type -> list of deltas
    paired_deltas = {agg: {t: [] for t in corruption_types} for agg in ("mean", "max", "slot_only")}
    
    for task in tasks:
        # Render clean trace
        clean_trace = render_trace(task)
        
        # Inject corruptions
        corruptions = inject_all_types(clean_trace, trace_rng)
        
        # Tokenize and score clean trace
        clean_ids, clean_spans = tokenize_with_provenance(tokenizer, clean_trace, device)
        clean_span_scores = retrospective_surprisal(model, clean_ids, clean_spans)
        
        # Map step_index -> SpanScore for clean trace (needed for twin pairing)
        clean_step_map = {s.step_index: s for s in clean_span_scores}
        
        # Store clean genuine scores
        for score in clean_span_scores:
            clean_genuine_scores["mean"].append(score.mean_neglogp)
            clean_genuine_scores["max"].append(score.max_neglogp)
            clean_genuine_scores["slot_only"].append(score.slot_only_neglogp)
            clean_genuine_surfaces.append(score.claim_surface)
            
        # Process each corruption type
        for corr_type in corruption_types:
            record = corruptions.get(corr_type)
            if record is None:
                continue
                
            # Tokenize and score corrupted trace
            corr_ids, corr_spans = tokenize_with_provenance(tokenizer, record.trace, device)
            corr_span_scores = retrospective_surprisal(model, corr_ids, corr_spans)
            
            corr_step = record.step_index
            
            for score in corr_span_scores:
                step = score.step_index
                
                if score.is_corrupted:
                    # Corrupted span
                    assert step == corr_step
                    for agg in ("mean", "max", "slot_only"):
                        val = getattr(score, f"{agg}_neglogp")
                        corrupt_scores[agg][corr_type].append(val)
                        
                        # Compute paired delta: corr - clean twin
                        if corr_step in clean_step_map:
                            twin_val = getattr(clean_step_map[corr_step], f"{agg}_neglogp")
                            paired_deltas[agg][corr_type].append(val - twin_val)
                elif step < corr_step:
                    # Before corruption
                    for agg in ("mean", "max", "slot_only"):
                        before_scores[agg][corr_type].append(getattr(score, f"{agg}_neglogp"))
                elif step > corr_step:
                    # Cascade (after corruption)
                    for agg in ("mean", "max", "slot_only"):
                        cascade_scores[agg][corr_type].append(getattr(score, f"{agg}_neglogp"))

    # 3. Analyze and build results
    agg_results = {}
    for agg in ("mean", "max", "slot_only"):
        res = AggregationResult()
        
        # Clean genuine average
        if clean_genuine_scores[agg]:
            res.mean_genuine_clean = float(np.mean(clean_genuine_scores[agg]))
            
        # Per-corruption statistics
        for corr_type in corruption_types:
            c_list = corrupt_scores[agg][corr_type]
            b_list = before_scores[agg][corr_type]
            cas_list = cascade_scores[agg][corr_type]
            pd_list = paired_deltas[agg][corr_type]
            
            # Means
            if c_list:
                res.mean_corrupt[corr_type] = float(np.mean(c_list))
            if b_list:
                res.mean_genuine_before[corr_type] = float(np.mean(b_list))
            if cas_list:
                res.mean_genuine_cascade[corr_type] = float(np.mean(cas_list))
            if pd_list:
                res.mean_paired_delta[corr_type] = float(np.mean(pd_list))
                
            # Pooled AUROC
            # Positive: corrupted spans
            # Negative: all clean genuine spans
            if c_list and clean_genuine_scores[agg]:
                scores_all = c_list + clean_genuine_scores[agg]
                labels_all = [1] * len(c_list) + [0] * len(clean_genuine_scores[agg])
                res.pooled_auroc[corr_type] = auroc(scores_all, labels_all)
            else:
                res.pooled_auroc[corr_type] = float("nan")
                
            # Matched-surface stratified AUROC (A2)
            # ghost / mislocation matches container-claims
            # phantom matches nowhere-claims
            matching_surface = "nowhere" if corr_type == "phantom" else "container"
            matched_genuine = [
                val for val, surf in zip(clean_genuine_scores[agg], clean_genuine_surfaces)
                if surf == matching_surface
            ]
            if c_list and matched_genuine:
                scores_match = c_list + matched_genuine
                labels_match = [1] * len(c_list) + [0] * len(matched_genuine)
                res.matched_surface_auroc[corr_type] = auroc(scores_match, labels_match)
            else:
                res.matched_surface_auroc[corr_type] = float("nan")
                
        agg_results[agg] = res

    wall_seconds = time.time() - start_time
    
    return ExperimentResultRecord(
        model_name=meta.model_name,
        task_label=task_cfg.label,
        seed=seed,
        n_traces=n_traces,
        mean=agg_results["mean"],
        max=agg_results["max"],
        slot_only=agg_results["slot_only"],
        meta=meta.to_dict(),
        wall_seconds=wall_seconds,
    )


def run_all_experiments() -> list[ExperimentResultRecord]:
    """Execute the full Phase 0 suite over configured models and tasks."""
    config = Phase0Config()
    
    # Create results folder
    os.makedirs(config.results_dir, exist_ok=True)
    results_file = Path(config.results_dir) / "llm_phase0_results.jsonl"
    
    all_records: list[ExperimentResultRecord] = []
    
    for model_name in config.model_names:
        print(f"\n======================================================================")
        print(f"Loading {model_name} on {config.device} ({config.dtype})...")
        print(f"======================================================================")
        
        try:
            model, tokenizer, meta = load_model(
                model_name,
                device=config.device,
                dtype=config.dtype,
                enable_thinking=config.enable_thinking,
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
            
        for task_cfg in config.task_configs:
            for seed in config.seeds:
                print(f"Running {task_cfg.label} task | seed {seed} | traces {config.n_traces}...")
                record = run_model_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    meta=meta,
                    task_cfg=task_cfg,
                    seed=seed,
                    n_traces=config.n_traces,
                    corruption_types=config.corruption_types,
                    device=config.device,
                )
                
                # Print summary
                print(f"  Completed in {record.wall_seconds:.1f}s.")
                for agg in ("mean", "max", "slot_only"):
                    res = getattr(record, agg)
                    print(f"    [{agg}] genuine_clean_avg = {res.mean_genuine_clean:.3f}")
                    for corr in config.corruption_types:
                        p_auroc = res.pooled_auroc.get(corr, float("nan"))
                        m_auroc = res.matched_surface_auroc.get(corr, float("nan"))
                        p_delta = res.mean_paired_delta.get(corr, float("nan"))
                        print(
                            f"      {corr:<12}: pooled_auroc={p_auroc:.3f} | "
                            f"matched_surface_auroc={m_auroc:.3f} | "
                            f"paired_delta={p_delta:+.3f}"
                        )
                
                # Save to JSONL
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record)) + "\n")
                    
                all_records.append(record)
                
        # Clean up memory before loading the next model
        print(f"Cleaning up {model_name} weights...")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    return all_records


if __name__ == "__main__":
    run_all_experiments()
