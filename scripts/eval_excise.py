"""Excised-lie eval: does removing the planted lie from attention repair the
answer, and does the hole-rehearsed (drop) model re-derive better than base?

Reuses the saved Phase 2 adapters (no retraining). For each model, reports
planted-error final-answer accuracy with the lie VISIBLE vs EXCISED (masked).
- excised >> visible  => excision-at-inference repairs (the Phase 3 core).
- drop.excised > base.excised => hole-rehearsal built operate-across-hole skill.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from source_monitor.llm.ood import entity_prose
from source_monitor.llm.train.config import Phase2Config
from source_monitor.llm.train.eval_repair import (
    load_for_eval,
    rank_accuracy,
    rank_accuracy_excised,
)

cfg = Phase2Config()
N = 200


def eval_one(adapter: str | None, seed: int):
    model, tok, _ = load_for_eval(cfg.model_name, cfg.device, cfg.dtype, adapter)
    planted = [t for t in entity_prose.generate(seed, N, corrupt_mid=True)
               if t.meta.get("corrupt_turn_index") is not None]
    vis = rank_accuracy(model, tok, planted, cfg.device)
    exc = rank_accuracy_excised(model, tok, planted, cfg.device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return vis, exc


def main():
    rows = {}
    plan = [("base_noft", None)]
    for arm in ("base", "drop"):
        plan.append((arm, arm))
    for tag, arm in plan:
        vis_s, exc_s = [], []
        for seed in cfg.seeds:
            adapter = None if arm is None else f"results/phase2_adapters/{arm}_seed{seed}"
            if adapter is not None and not Path(adapter).exists():
                continue
            vis, exc = eval_one(adapter, seed)
            vis_s.append(vis)
            exc_s.append(exc)
            print(f"  {tag:<10} seed{seed}: visible={vis:.3f}  excised={exc:.3f}", flush=True)
        if vis_s:
            rows[tag] = (np.mean(vis_s), np.mean(exc_s))
    print("\n=== planted-error final-answer accuracy (mean over seeds) ===")
    print(f"{'model':<10} {'lie VISIBLE':<14} {'lie EXCISED':<14} {'excision gain'}")
    for tag, (v, e) in rows.items():
        print(f"{tag:<10} {v:<14.3f} {e:<14.3f} {e - v:+.3f}")
    if "base" in rows and "drop" in rows:
        print(f"\ndrop vs base under excision: {rows['drop'][1]:.3f} vs {rows['base'][1]:.3f} "
              f"(Δ{rows['drop'][1] - rows['base'][1]:+.3f})")


if __name__ == "__main__":
    main()
