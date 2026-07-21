"""F23 runner: does ensemble-averaged surprisal beat single-pass? (tests F22c)

This is a DETECTOR-level study, deliberately not a full Phase 3 re-run. It scores
spans and measures separability; it never generates. That makes it ~10x cheaper
than the closed loop, so the sweep can be wide, and it isolates the one thing
under test — the quality of the surprisal estimate — from the repair machinery
that F21 already validated.

Only if a config clears H-ens-1/H-ens-2 is it worth paying for a full
``loop/phase3.py`` run with ensemble scoring wired in.

Pre-registered (F22):
  H-ens-1  AUROC(ensemble) - AUROC(single) >= +.02
  H-ens-2  at clean false-excision <= .02, catch rate gain >= +.05
  H-ens-3  rank 8 within .01 AUROC of rank 64
  control  sigma=0 must reproduce single-pass EXACTLY

Run:
    python -m source_monitor.llm.loop.f22_ensemble --quick     # smoke test
    python -m source_monitor.llm.loop.f22_ensemble             # full sweep

GPU HEAT: this is a long run of back-to-back forward passes on a card with a
broken fan. Start the floor fans and watch ``nvidia-smi`` before the full sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.ensemble import ensemble_span_scores
from source_monitor.llm.loop.monitor import build_context
from source_monitor.llm.ood import entity_prose


@dataclass(frozen=True)
class Arm:
    """One scoring configuration under test."""

    label: str
    k: int
    sigma: float
    rank: int = 8


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties counted as half.

    Args:
        pos: Scores of corrupted spans.
        neg: Scores of genuine spans.

    Returns:
        Probability a random positive outranks a random negative; nan if either
        class is empty.
    """
    if not pos or not neg:
        return float("nan")
    allv = np.concatenate([np.asarray(pos), np.asarray(neg)])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within tie groups
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _score_traces(
    model, tok, traces, device: str, arm: Arm, base_seed: int
) -> list[dict]:
    """Score every assistant span of every trace under one arm.

    Returns:
        One record per trace: span scores, ensemble dispersion, and the index of
        the corrupted span (``None`` for clean traces).
    """
    out: list[dict] = []
    for tr in traces:
        ids, spans, asst = build_context(tok, tr, device)
        mean, std = ensemble_span_scores(
            model, ids, spans, k=arm.k, sigma=arm.sigma,
            rank=arm.rank, base_seed=base_seed,
        )
        if len(mean) != len(asst):  # provenance/telemetry span filters diverged
            raise RuntimeError(
                f"span/score length mismatch: {len(mean)} scores vs {len(asst)} spans"
            )
        ci = tr.meta.get("corrupt_turn_index")
        corrupt_i = next((i for i, (ti, _s, _e) in enumerate(asst) if ti == ci), None)
        out.append({"scores": mean, "std": std, "corrupt_i": corrupt_i})
    return out


def _evaluate(
    planted: list[dict],
    clean: list[dict],
    calib: list[dict],
    target_fpr: float,
) -> dict:
    """Turn scored traces into the pre-registered metrics.

    The floor is set from a HELD-OUT clean set's per-trace maximum at the
    ``1 - target_fpr`` quantile, so every arm is compared at the same nominal
    idle false-excision rate. Comparing catch rates at a shared floor in nats
    would be meaningless — perturbation shifts the whole scale, so the floor has
    to be re-derived per arm.

    ``calib`` must come from a seed disjoint from the eval seeds, matching F21e's
    discipline (``calib_seed=7``). Calibrating and measuring ``clean_fp_rate`` on
    the same traces makes that rate true by construction and quietly fits the
    floor to the eval data — the first thing a reader would object to.
    """
    pos = [t["scores"][t["corrupt_i"]] for t in planted if t["corrupt_i"] is not None]
    neg = [
        s
        for t in planted
        for i, s in enumerate(t["scores"])
        if i != t["corrupt_i"]
    ] + [s for t in clean for s in t["scores"]]

    calib_max = [max(t["scores"]) for t in calib if t["scores"]]
    floor = (
        float(np.quantile(calib_max, 1.0 - target_fpr)) if calib_max else float("inf")
    )

    # Trace-level: the monitor flags argmax, then requires it to clear the floor.
    catch = n = 0
    for t in planted:
        if t["corrupt_i"] is None or not t["scores"]:
            continue
        n += 1
        best = int(np.argmax(t["scores"]))
        if best == t["corrupt_i"] and t["scores"][best] > floor:
            catch += 1

    fp = sum(1 for t in clean if t["scores"] and max(t["scores"]) > floor)
    all_std = [s for t in planted + clean for s in t["std"]]

    return {
        "auroc": auroc(pos, neg),
        "catch_rate": catch / n if n else float("nan"),
        "clean_fp_rate": fp / len(clean) if clean else float("nan"),
        "floor": floor,
        "mean_member_std": float(np.mean(all_std)) if all_std else 0.0,
        "n_planted": n,
        "n_clean": len(clean),
        "n_calib": len(calib),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="F23: ensemble vs single-pass detector")
    ap.add_argument("--quick", action="store_true",
                    help="6 traces, 1 seed, 3 arms — a smoke test, not evidence")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-traces", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--confound", action="store_true",
                    help="F23d: hold sigma=0.05, sweep k in {1,2,4,8} to test "
                         "whether the F23 win is ENSEMBLE-averaging or a single "
                         "perturbed pass. k1==single-perturbed isolates it.")
    args = ap.parse_args()

    cfg = Phase3Config()
    model_name = args.model or cfg.model_name
    seeds = tuple(args.seeds) if args.seeds else ((42,) if args.quick else cfg.seeds)
    n_traces = args.n_traces or (6 if args.quick else 60)
    # A q=.98 floor off 8 traces is meaningless; --quick only checks the plumbing.
    calib_n = 8 if args.quick else cfg.calib_n
    results_dir = args.results_dir or cfg.results_dir

    if args.confound:
        # F23d: sigma fixed at the winning 0.05, k varied. If k1 ≈ k8, the F23
        # gain is the PERTURBATION, not the ensemble, and the eBP framing drops.
        arms = [
            Arm("single", k=1, sigma=0.0),  # baseline anchor for the deltas
            Arm("k1-s0.05", k=1, sigma=0.05),
            Arm("k2-s0.05", k=2, sigma=0.05),
            Arm("k4-s0.05", k=4, sigma=0.05),
            Arm("k8-s0.05", k=8, sigma=0.05),
        ]
    else:
        arms = [
            Arm("single", k=1, sigma=0.0),
            Arm("control-k4-s0", k=4, sigma=0.0),  # must equal single: harness check
            Arm("k4-s0.01", k=4, sigma=0.01),
        ]
        if not args.quick:
            arms += [
                Arm("k4-s0.005", k=4, sigma=0.005),
                Arm("k4-s0.02", k=4, sigma=0.02),
                Arm("k4-s0.05", k=4, sigma=0.05),
                Arm("k8-s0.01", k=8, sigma=0.01),
                Arm("k8-s0.02", k=8, sigma=0.02),
                Arm("k4-s0.02-r64", k=4, sigma=0.02, rank=64),  # H-ens-3
            ]

    os.makedirs(results_dir, exist_ok=True)
    out_file = Path(results_dir) / "llm_f23_ensemble_results.jsonl"

    print(f"model={model_name} seeds={seeds} n_traces={n_traces} "
          f"arms={len(arms)} target_fpr={args.target_fpr}")
    if not args.quick:
        print("GPU HEAT WARNING: sustained forward passes. Start the floor fans.")

    model, tok, _meta = load_model(model_name, device=cfg.device, dtype=cfg.dtype,
                                   enable_thinking=False)
    model.eval()

    rows: list[dict] = []
    for arm in arms:
        t0 = time.time()
        # Held-out calibration set (cfg.calib_seed=7, disjoint from eval seeds).
        # Scored once per arm, reused across eval seeds.
        calib_tr = entity_prose.generate(cfg.calib_seed, calib_n)
        calib = _score_traces(model, tok, calib_tr, cfg.device, arm, cfg.calib_seed)

        per_seed: list[dict] = []
        for seed in seeds:
            planted_tr = [
                t for t in entity_prose.generate(seed, n_traces, corrupt_mid=True)
                if t.meta.get("corrupt_turn_index") is not None
            ]
            clean_tr = entity_prose.generate(seed, n_traces)
            planted = _score_traces(model, tok, planted_tr, cfg.device, arm, seed)
            clean = _score_traces(model, tok, clean_tr, cfg.device, arm, seed + 1000)
            per_seed.append(_evaluate(planted, clean, calib, args.target_fpr))

        agg = {
            key: float(np.mean([p[key] for p in per_seed]))
            for key in ("auroc", "catch_rate", "clean_fp_rate", "floor",
                        "mean_member_std")
        }
        rec = {"arm": arm.label, "k": arm.k, "sigma": arm.sigma, "rank": arm.rank,
               "model_name": model_name, "seeds": list(seeds), "n_traces": n_traces,
               "calib_n": calib_n, "calib_seed": cfg.calib_seed,
               "target_fpr": args.target_fpr,
               "wall_s": round(time.time() - t0, 1), **agg}
        rows.append(rec)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {arm.label:<16} auroc={agg['auroc']:.4f} "
              f"catch={agg['catch_rate']:.3f} cleanFP={agg['clean_fp_rate']:.3f} "
              f"floor={agg['floor']:.2f} memberSD={agg['mean_member_std']:.4f} "
              f"({rec['wall_s']}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    base = next(r for r in rows if r["arm"] == "single")
    ctrl = next((r for r in rows if r["arm"] == "control-k4-s0"), None)

    print("\n" + "=" * 78)
    print(f"{'arm':<18}{'auroc':<10}{'Δauroc':<10}{'catch':<9}{'Δcatch':<9}"
          f"{'cleanFP':<9}memberSD")
    for r in rows:
        print(f"{r['arm']:<18}{r['auroc']:<10.4f}{r['auroc']-base['auroc']:<+10.4f}"
              f"{r['catch_rate']:<9.3f}{r['catch_rate']-base['catch_rate']:<+9.3f}"
              f"{r['clean_fp_rate']:<9.3f}{r['mean_member_std']:.4f}")

    print("\nGATE")
    if ctrl is not None:
        ok = (abs(ctrl["auroc"] - base["auroc"]) < 1e-9
              and ctrl["mean_member_std"] < 1e-9)
        print(f"control  sigma=0 reproduces single-pass exactly: "
              f"Δauroc={ctrl['auroc']-base['auroc']:+.2e} "
              f"memberSD={ctrl['mean_member_std']:.2e}  "
              f"[{'PASS' if ok else 'FAIL — harness bug, ignore all other rows'}]")

    live = [r for r in rows if r["sigma"] > 0]
    best_a = max(live, key=lambda r: r["auroc"]) if live else None
    best_c = max(live, key=lambda r: r["catch_rate"]) if live else None
    if best_a and best_c:
        d_a = best_a["auroc"] - base["auroc"]
        d_c = best_c["catch_rate"] - base["catch_rate"]
        print(f"H-ens-1  best Δauroc >= +.02: {best_a['arm']} {d_a:+.4f}  "
              f"[{'PASS' if d_a >= 0.02 else 'FAIL'}]")
        print(f"H-ens-2  best Δcatch >= +.05 at cleanFP<={args.target_fpr}: "
              f"{best_c['arm']} {d_c:+.3f} (cleanFP={best_c['clean_fp_rate']:.3f})  "
              f"[{'PASS' if d_c >= 0.05 else 'FAIL'}]")
    r8 = next((r for r in rows if r["arm"] == "k4-s0.02"), None)
    r64 = next((r for r in rows if r["arm"] == "k4-s0.02-r64"), None)
    if r8 and r64:
        d = abs(r64["auroc"] - r8["auroc"])
        print(f"H-ens-3  rank8 within .01 auroc of rank64: |Δ|={d:.4f}  "
              f"[{'PASS' if d <= 0.01 else 'FAIL'}]")

    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
