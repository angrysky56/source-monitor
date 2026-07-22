"""F24 runner: does paraphrase-INSTABILITY catch lies the single pass misses?

Tests the "murmuration of a claim" idea (F22/F23 arc). For every assistant span
we compute, over a faithful paraphrase family (paraphrase.py), the family MEAN
and STD of its slot surprisal, then ask whether the dispersion (std) is a
detector signal — and, the load-bearing question, whether it adds anything the
single-pass mean doesn't already have.

Detectors, all read off ONE family computation per trace (cheap to add):
    single     single-pass slot surprisal            (baseline; ~F21e .733)
    fam_mean   mean over the paraphrase family       (variance-reduced surprisal)
    fam_std    dispersion over the family, alone      (the NEW signal)
    fam_c0.5   fam_mean + 0.5 * fam_std               (combined)
    fam_c1.0   fam_mean + 1.0 * fam_std               (combined)

Pre-registered (record BEFORE reading results; F21b/F23 discipline):
    control  identity family -> fam_std == 0, fam_mean == single (harness gate)
    H-mur-1  AUROC(fam_std) >= .60          (dispersion carries ANY signal)
    H-mur-2  best fam_c catch - single >= +.05  AND  - fam_mean >= +.02
             (dispersion adds BEYOND the mean — the load-bearing test)

HONEST PRIOR (write it down now): H-mur-1 probably passes weakly — a surprising
span tends also to be an unstable one, so std correlates with mean. That very
correlation puts H-mur-2 at real risk of FAILING: if dispersion is just
surprising-ness in disguise, the +.02-over-mean margin won't appear. Either
outcome is informative; a null on H-mur-2 is the finding that paraphrase
instability is redundant with surprisal, not that the idea is unmeasurable.

Reuses F23's evaluation (auroc, _evaluate) verbatim, so numbers are directly
comparable to the F23 table.

Run:  python -m source_monitor.llm.loop.f24_murmuration --quick   # smoke
      python -m source_monitor.llm.loop.f24_murmuration            # full
GPU HEAT: sustained forward passes (m*k per trace). Fans on; watch nvidia-smi.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.f22_ensemble import _evaluate
from source_monitor.llm.loop.monitor import build_context, span_scores
from source_monitor.llm.loop.paraphrase import (
    family_span_stats,
    rerender_arithmetic,
    rerender_entity_prose,
)
from source_monitor.llm.ood import arithmetic, entity_prose

DETECTORS = ("single", "fam_mean", "fam_std", "fam_c0.5", "fam_c1.0")

# task -> (generator module, its faithful reworder). entity_prose is short-ξ (the
# answer is in the last turn — F20d) and single-pass already ~ceilings it; arithmetic
# is long-ξ (running total), where single-pass leaves headroom and support paraphrase
# should bite — the F24d test of whether dispersion is a DISTINCT axis.
TASKS = {
    "entity_prose": (entity_prose, rerender_entity_prose),
    "arithmetic": (arithmetic, rerender_arithmetic),
}


def _detector_scores(rec: dict, name: str) -> list[float]:
    """Extract one detector's per-span score list from a trace's family record."""
    if name == "single":
        return rec["single"]
    if name == "fam_mean":
        return rec["fmean"]
    if name == "fam_std":
        return rec["fstd"]
    lam = float(name.split("c")[1])
    return [m + lam * s for m, s in zip(rec["fmean"], rec["fstd"], strict=True)]


@torch.no_grad()
def _score_trace(model, tok, tr, device: str, k: int, seed: int, rerender_fn) -> dict:
    """Single-pass + family (mean, std) for every assistant span of one trace."""
    ids, spans, asst = build_context(tok, tr, device)
    single = span_scores(model, ids, spans)
    fmean, fstd, _asst = family_span_stats(
        model, tok, tr, device, k=k, base_seed=seed, rerender_fn=rerender_fn
    )
    if not (len(single) == len(fmean) == len(fstd)):
        raise RuntimeError("single/family span-count mismatch")
    ci = tr.meta.get("corrupt_turn_index")
    corrupt_i = next((a for a, (ti, _s, _e) in enumerate(asst) if ti == ci), None)
    return {"single": single, "fmean": fmean, "fstd": fstd, "corrupt_i": corrupt_i}


def _eval_detector(planted, clean, calib, name: str, fpr: float) -> dict:
    """Run F23's _evaluate on one detector's scores."""
    def recs(rs):
        return [
            {
                "scores": _detector_scores(r, name),
                "std": [0.0] * len(r["single"]),
                "corrupt_i": r["corrupt_i"],
            }
            for r in rs
        ]

    return _evaluate(recs(planted), recs(clean), recs(calib), fpr)


def _control_check(model, tok, traces, device: str, k: int) -> tuple[float, float]:
    """Identity family must give std==0 and mean==single. Returns worst deviations."""
    max_std = 0.0
    max_mean_dev = 0.0
    for tr in traces:
        ids, spans, _asst = build_context(tok, tr, device)
        single = span_scores(model, ids, spans)
        fmean, fstd, _ = family_span_stats(
            model, tok, tr, device, k=k, base_seed=0, identity=True
        )
        max_std = max(max_std, max((abs(s) for s in fstd), default=0.0))
        max_mean_dev = max(
            max_mean_dev,
            max((abs(a - b) for a, b in zip(fmean, single, strict=True)), default=0.0),
        )
    return max_std, max_mean_dev


def main() -> None:
    ap = argparse.ArgumentParser(description="F24: paraphrase-murmuration detector")
    ap.add_argument("--quick", action="store_true", help="6 traces, 1 seed, k=3 — smoke")
    ap.add_argument("--task", default="entity_prose", choices=list(TASKS),
                    help="entity_prose (short-ξ, ceilinged) or arithmetic (long-ξ, F24d)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-traces", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-ops", type=int, default=None,
                    help="chain length; longer = harder / more headroom (arithmetic)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = Phase3Config()
    model_name = args.model or cfg.model_name
    seeds = tuple(args.seeds) if args.seeds else ((42,) if args.quick else (42, 137))
    n_traces = args.n_traces or (6 if args.quick else 30)
    k = args.k or (3 if args.quick else 6)
    calib_n = 6 if args.quick else 20
    results_dir = args.results_dir or cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)
    mod, rerender_fn = TASKS[args.task]
    gkw = {} if args.n_ops is None else {"n_ops": args.n_ops}
    out_file = Path(results_dir) / f"llm_f24_{args.task}_results.jsonl"

    print(f"task={args.task} model={model_name} seeds={seeds} n_traces={n_traces} "
          f"k={k} n_ops={args.n_ops or 'default'} target_fpr={args.target_fpr}")
    if not args.quick:
        print("GPU HEAT: m*k forward passes per trace. Fans on; watch nvidia-smi.")

    model, tok, _meta = load_model(model_name, device=cfg.device, dtype=cfg.dtype,
                                   enable_thinking=False)
    model.eval()

    # Harness gate: identity family must be a true no-op.
    calib0 = mod.generate(cfg.calib_seed, max(4, calib_n // 4), **gkw)
    c_std, c_mean = _control_check(model, tok, calib0[:4], cfg.device, k=3)
    control_ok = c_std < 1e-6 and c_mean < 1e-6
    print(f"control (identity family): max_std={c_std:.2e} max|mean-single|={c_mean:.2e}"
          f"  [{'PASS' if control_ok else 'FAIL — harness bug'}]", flush=True)

    per_seed: dict[str, list[dict]] = {d: [] for d in DETECTORS}
    for seed in seeds:
        t0 = time.time()
        planted_tr = [t for t in mod.generate(seed, n_traces, corrupt_mid=True, **gkw)
                      if t.meta.get("corrupt_turn_index") is not None]
        clean_tr = mod.generate(seed, n_traces, **gkw)
        calib_tr = mod.generate(cfg.calib_seed, calib_n, **gkw)

        planted = [_score_trace(model, tok, t, cfg.device, k, seed, rerender_fn)
                   for t in planted_tr]
        clean = [_score_trace(model, tok, t, cfg.device, k, seed + 1000, rerender_fn)
                 for t in clean_tr]
        calib = [_score_trace(model, tok, t, cfg.device, k, cfg.calib_seed, rerender_fn)
                 for t in calib_tr]

        for d in DETECTORS:
            per_seed[d].append(_eval_detector(planted, clean, calib, d, args.target_fpr))
        print(f"  seed {seed}: scored {len(planted)}+{len(clean)}+{len(calib)} traces "
              f"({time.time() - t0:.0f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    def agg(d, key):
        return float(np.mean([m[key] for m in per_seed[d]]))

    rows = []
    for d in DETECTORS:
        rec = {"detector": d, "task": args.task, "model_name": model_name,
               "seeds": list(seeds), "n_traces": n_traces, "k": k,
               "target_fpr": args.target_fpr,
               "auroc": agg(d, "auroc"), "catch_rate": agg(d, "catch_rate"),
               "clean_fp_rate": agg(d, "clean_fp_rate")}
        rows.append(rec)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    base = next(r for r in rows if r["detector"] == "single")
    fmean_r = next(r for r in rows if r["detector"] == "fam_mean")
    fstd_r = next(r for r in rows if r["detector"] == "fam_std")
    combs = [r for r in rows if r["detector"].startswith("fam_c")]
    best_c = max(combs, key=lambda r: r["catch_rate"])

    print("\n" + "=" * 70)
    print(f"{'detector':<12}{'auroc':<10}{'catch':<9}{'Δcatch':<9}{'cleanFP'}")
    for r in rows:
        print(f"{r['detector']:<12}{r['auroc']:<10.4f}{r['catch_rate']:<9.3f}"
              f"{r['catch_rate'] - base['catch_rate']:<+9.3f}{r['clean_fp_rate']:.3f}")

    print("\nGATE")
    print(f"control  identity family no-op: "
          f"[{'PASS' if control_ok else 'FAIL'}]")
    print(f"H-mur-1  AUROC(fam_std) >= .60: {fstd_r['auroc']:.4f}  "
          f"[{'PASS' if fstd_r['auroc'] >= 0.60 else 'FAIL'}]")
    d_single = best_c["catch_rate"] - base["catch_rate"]
    d_mean = best_c["catch_rate"] - fmean_r["catch_rate"]
    h2 = d_single >= 0.05 and d_mean >= 0.02
    print(f"H-mur-2  best {best_c['detector']} catch {best_c['catch_rate']:.3f}: "
          f"vs single {d_single:+.3f} (>=+.05), vs fam_mean {d_mean:+.3f} (>=+.02)  "
          f"[{'PASS' if h2 else 'FAIL'}]")
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
