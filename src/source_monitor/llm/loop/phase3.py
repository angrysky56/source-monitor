"""Phase 3 runner: the closed loop on free generation.

Compares monitor_off / monitor_on / oracle_excise on planted-lie traces, and
monitor_off / monitor_on on CLEAN traces (the idle cost). Writes flat records to
results/llm_phase3_results.jsonl and prints the pre-registered gate.

Run: python -m source_monitor.llm.loop.phase3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.monitor import calibrate_floor, run_case
from source_monitor.llm.ood import entity_prose


def _agg(cases: list[dict]) -> dict:
    n = max(len(cases), 1)
    excised = [c for c in cases if c["excised"]]
    return {
        "n": len(cases),
        "accuracy": sum(c["correct"] for c in cases) / n,
        "abstain_rate": sum(c["abstain"] for c in cases) / n,
        "excise_rate": len(excised) / n,
        "flag_hit_rate": (sum(c["flag_hit"] for c in excised) / len(excised)) if excised else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 closed-loop source monitor")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-traces", type=int, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--flag-mode", default=None, choices=["absolute", "zscore", "both"])
    args = ap.parse_args()

    cfg = Phase3Config()
    cfg = replace(
        cfg,
        seeds=tuple(args.seeds) if args.seeds else cfg.seeds,
        n_traces=args.n_traces or cfg.n_traces,
        k_threshold=args.k if args.k is not None else cfg.k_threshold,
        model_name=args.model or cfg.model_name,
        flag_mode=args.flag_mode or cfg.flag_mode,
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_file = Path(cfg.results_dir) / "llm_phase3_results.jsonl"

    model, tok, meta = load_model(cfg.model_name, device=cfg.device, dtype=cfg.dtype,
                                  enable_thinking=False)
    model.eval()

    # Absolute floor calibrated on CLEAN traces held out from the eval seeds.
    floor = None
    if cfg.flag_mode in ("absolute", "both"):
        calib = entity_prose.generate(cfg.calib_seed, cfg.calib_n)
        floor = calibrate_floor(model, tok, calib, cfg.device, cfg.calib_quantile)
        print(f"calibrated floor (q={cfg.calib_quantile}, n={cfg.calib_n}, "
              f"seed={cfg.calib_seed}): {floor:.3f}  [mode={cfg.flag_mode}]", flush=True)

    summary: dict[tuple[str, str], list[dict]] = {}
    for seed in cfg.seeds:
        planted = [t for t in entity_prose.generate(seed, cfg.n_traces, corrupt_mid=True)
                   if t.meta.get("corrupt_turn_index") is not None]
        clean = entity_prose.generate(seed, cfg.n_traces)

        for split, traces, conds in (
            ("planted", planted, cfg.conditions),
            ("clean", clean, ("monitor_off", "monitor_on")),
        ):
            for cond in conds:
                t0 = time.time()
                cases = [run_case(model, tok, tr, cfg, cond, floor=floor) for tr in traces]
                a = _agg(cases)
                rec = {"split": split, "condition": cond, "seed": seed,
                       "model_name": cfg.model_name, "k": cfg.k_threshold,
                       "flag_mode": cfg.flag_mode, "floor": floor,
                       "wall_s": round(time.time() - t0, 1), **a}
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                summary.setdefault((split, cond), []).append(a)
                print(f"  {split:<8} {cond:<14} seed{seed}: acc={a['accuracy']:.3f} "
                      f"excise={a['excise_rate']:.3f} hit={a['flag_hit_rate']:.3f} "
                      f"abstain={a['abstain_rate']:.3f}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    def m(split, cond, key):
        rows = summary.get((split, cond), [])
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    print("\n" + "=" * 78)
    print(f"{'split':<10}{'condition':<16}{'accuracy':<12}{'excise':<10}{'flag hit':<10}abstain")
    for (split, cond) in summary:
        print(f"{split:<10}{cond:<16}{m(split,cond,'accuracy'):<12.3f}"
              f"{m(split,cond,'excise_rate'):<10.3f}{m(split,cond,'flag_hit_rate'):<10.3f}"
              f"{m(split,cond,'abstain_rate'):.3f}")

    print("\nGATE")
    off = m("planted", "monitor_off", "accuracy")
    on = m("planted", "monitor_on", "accuracy")
    orc = m("planted", "oracle_excise", "accuracy")
    hit = m("planted", "monitor_on", "flag_hit_rate")
    c_off = m("clean", "monitor_off", "accuracy")
    c_on = m("clean", "monitor_on", "accuracy")
    c_exc = m("clean", "monitor_on", "excise_rate")
    print(f"P-3.1 loop repairs (on - off >= .15): {off:.3f} -> {on:.3f} "
          f"(Δ{on-off:+.3f})  [{'PASS' if on - off >= 0.15 else 'FAIL'}]")
    print(f"P-3.2 detector not the bottleneck (hit >= .80, on within .05 of oracle): "
          f"hit={hit:.3f} oracle={orc:.3f}  "
          f"[{'PASS' if (hit >= 0.80 and abs(orc - on) <= 0.05) else 'FAIL'}]")
    print(f"P-3.3 cheap when idle (clean excise <= .10, acc drop <= .02): "
          f"excise={c_exc:.3f} acc {c_off:.3f} -> {c_on:.3f}  "
          f"[{'PASS' if (c_exc <= 0.10 and c_off - c_on <= 0.02) else 'FAIL'}]")


if __name__ == "__main__":
    main()
