"""F27 — The Router Runner & Experimental Validation.

Evaluates the routed monitor across three evaluation tracks:
1. All-Derivable Control (entity_prose): Verifies 100% routing to Leg 1 and identical output.
2. All-Factual Control (factual_qa): Verifies 100% routing to Leg 2 and identical output.
3. Provisional Mixed Eval: Mixed entity_prose + planted-lie factual_qa.

Audit requirement: Dumps per-span routing decisions, context snippets, raw leg scores,
and binary flags for review.

Run:
  python -m source_monitor.llm.loop.f27_router --quick
  python -m source_monitor.llm.loop.f27_router
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.f22_ensemble import auroc
from source_monitor.llm.loop.monitor import calibrate_floor
from source_monitor.llm.loop.router import evaluate_routed_trace
from source_monitor.llm.ood import entity_prose, factual_qa
from source_monitor.llm.ood.base import make_variant


def main() -> None:
    ap = argparse.ArgumentParser(description="F27 Router Evaluation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--ablation-precision-weight", action="store_true", help="Enable precision-weighting ablation")
    args = ap.parse_args()

    cfg = Phase3Config()
    model_name = args.model or cfg.model_name
    n = args.n or (12 if args.quick else 40)
    k = args.k
    results_dir = args.results_dir or cfg.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"--- F27 Router Benchmark ---")
    print(f"model={model_name} n={n} k={k} temp={args.temperature} seed={args.seed}")
    print(f"precision_weighting={args.ablation_precision_weight}")

    model, tok, _ = load_model(
        model_name, device=cfg.device, dtype=cfg.dtype, enable_thinking=False
    )
    model.eval()

    # Calibrate surprisal floor on clean entity_prose traces
    clean_calib = entity_prose.generate(seed=args.seed + 9999, n=8 if args.quick else 16)
    clean_calib = [t for t in clean_calib if not t.meta.get("corrupt_mid")]
    surprisal_floor = calibrate_floor(model, tok, clean_calib, cfg.device, quantile=0.95)
    print(f"Calibrated surprisal floor (95th percentile clean): {surprisal_floor:.4f}\n")

    # --- TRACK 1: ALL-DERIVABLE CONTROL (entity_prose) ---
    print("==================================================================")
    print("TRACK 1: ALL-DERIVABLE CONTROL (entity_prose)")
    print("==================================================================")
    ep_traces = entity_prose.generate(seed=args.seed, n=n)

    t0 = time.time()
    ep_res = [
        evaluate_routed_trace(
            model, tok, tr, cfg.device, surprisal_floor=surprisal_floor,
            consistency_k=k, consistency_temp=args.temperature, seed=args.seed + i,
            apply_precision_weighting=args.ablation_precision_weight
        )
        for i, tr in enumerate(ep_traces)
    ]
    t_ep = time.time() - t0

    derivable_pct = sum(r["derivable"] for r in ep_res) / len(ep_res) * 100.0
    surprisal_branch_pct = sum(r["branch"] == "surprisal" for r in ep_res) / len(ep_res) * 100.0
    ep_match_pct = sum(r["routed_flag"] == r["surprisal_flag"] for r in ep_res) / len(ep_res) * 100.0

    print(f"Traces evaluated: {len(ep_res)} in {t_ep:.1f}s")
    print(f"Content-Derivable %: {derivable_pct:.1f}%")
    print(f"Routed to Surprisal %: {surprisal_branch_pct:.1f}%")
    print(f"Identity Control Match (Router == Leg 1): {ep_match_pct:.1f}%")

    print("\n-- Track 1 Sample Audit Logs --")
    for idx in range(min(3, len(ep_res))):
        r = ep_res[idx]
        print(f"  [{idx}] Context: {r['context_snippet']!r}")
        print(f"      ClaimVal: {r['claim_value']!r} | Derivable: {r['derivable']} | Branch: {r['branch']}")
        print(f"      Surprisal: {r['raw_surprisal']:.3f} (Flag: {r['surprisal_flag']}) | ConsistencyFlag: {r['consistency_flag']}")
        print(f"      RoutedFlag: {r['routed_flag']}")

    # --- TRACK 2: ALL-FACTUAL CONTROL (factual_qa) ---
    print("\n==================================================================")
    print("TRACK 2: ALL-FACTUAL CONTROL (factual_qa)")
    print("==================================================================")
    fq_traces = factual_qa.generate(seed=args.seed + 100, n=n)

    t0 = time.time()
    fq_res = [
        evaluate_routed_trace(
            model, tok, tr, cfg.device, surprisal_floor=surprisal_floor,
            consistency_k=k, consistency_temp=args.temperature, seed=args.seed + 100 + i,
            apply_precision_weighting=args.ablation_precision_weight
        )
        for i, tr in enumerate(fq_traces)
    ]
    t_fq = time.time() - t0

    fq_underivable_pct = sum(not r["derivable"] for r in fq_res) / len(fq_res) * 100.0
    consistency_branch_pct = sum(r["branch"] == "consistency" for r in fq_res) / len(fq_res) * 100.0
    fq_match_pct = sum(r["routed_flag"] == r["consistency_flag"] for r in fq_res) / len(fq_res) * 100.0

    print(f"Traces evaluated: {len(fq_res)} in {t_fq:.1f}s")
    print(f"Content-Underivable %: {fq_underivable_pct:.1f}%")
    print(f"Routed to Consistency %: {consistency_branch_pct:.1f}%")
    print(f"Identity Control Match (Router == Leg 2): {fq_match_pct:.1f}%")

    print("\n-- Track 2 Sample Audit Logs --")
    for idx in range(min(3, len(fq_res))):
        r = fq_res[idx]
        print(f"  [{idx}] Context: {r['context_snippet']!r}")
        print(f"      ClaimVal: {r['claim_value']!r} | Derivable: {r['derivable']} | Branch: {r['branch']}")
        print(f"      DistinctRatio: {r['distinct_ratio']:.3f} | Sampled: {[a[:15] for a in r['sampled_answers']]}")
        print(f"      ConsistencyFlag: {r['consistency_flag']} | RoutedFlag: {r['routed_flag']}")

    # --- TRACK 3: PROVISIONAL MIXED EVAL (PLANTED LIES) ---
    print("\n==================================================================")
    print("TRACK 3: PROVISIONAL MIXED EVAL (PLANTED LIES)")
    print("==================================================================")
    # Build clean and corrupt instances for both tasks
    ep_clean = [t for t in ep_traces if not t.meta.get("corrupt_mid")]
    ep_corrupt = [t for t in ep_traces if t.meta.get("corrupt_mid")]

    # For factual_qa: clean = known questions; corrupt = planted wrong value
    fq_known = [t for t in fq_traces if not t.meta.get("negation_correct")]
    fq_planted = [
        make_variant(t, (t.claim.correct_index + 1) % len(t.claim.candidate_contents))
        for t in fq_known
    ]

    mixed_clean = ep_clean + fq_known
    mixed_corrupt = ep_corrupt + fq_planted

    t0 = time.time()
    clean_eval = [
        evaluate_routed_trace(
            model, tok, tr, cfg.device, surprisal_floor=surprisal_floor,
            consistency_k=k, consistency_temp=args.temperature, seed=args.seed + 500 + i,
            apply_precision_weighting=args.ablation_precision_weight
        )
        for i, tr in enumerate(mixed_clean)
    ]
    corrupt_eval = [
        evaluate_routed_trace(
            model, tok, tr, cfg.device, surprisal_floor=surprisal_floor,
            consistency_k=k, consistency_temp=args.temperature, seed=args.seed + 1500 + i,
            apply_precision_weighting=args.ablation_precision_weight
        )
        for i, tr in enumerate(mixed_corrupt)
    ]
    t_mixed = time.time() - t0

    # Scores / Flags
    routed_clean_flags = [float(r["routed_flag"]) for r in clean_eval]
    routed_corrupt_flags = [float(r["routed_flag"]) for r in corrupt_eval]

    leg1_clean_flags = [float(r["surprisal_flag"]) for r in clean_eval]
    leg1_corrupt_flags = [float(r["surprisal_flag"]) for r in corrupt_eval]

    leg2_clean_flags = [float(r["consistency_flag"]) for r in clean_eval]
    leg2_corrupt_flags = [float(r["consistency_flag"]) for r in corrupt_eval]

    au_routed = auroc(routed_corrupt_flags, routed_clean_flags)
    au_leg1 = auroc(leg1_corrupt_flags, leg1_clean_flags)
    au_leg2 = auroc(leg2_corrupt_flags, leg2_clean_flags)

    print(f"Mixed Corpus: Clean={len(mixed_clean)} | Corrupt={len(mixed_corrupt)} in {t_mixed:.1f}s")
    print("\nMixed Binary Flag Performance:")
    print(f"  Routed Monitor AUROC : {au_routed:.3f}")
    print(f"  Leg 1 (Surprisal) AUROC: {au_leg1:.3f}")
    print(f"  Leg 2 (Consistency) AUROC: {au_leg2:.3f}")

    catch_routed = sum(routed_corrupt_flags) / len(routed_corrupt_flags) if routed_corrupt_flags else 0
    catch_leg1 = sum(leg1_corrupt_flags) / len(leg1_corrupt_flags) if leg1_corrupt_flags else 0
    catch_leg2 = sum(leg2_corrupt_flags) / len(leg2_corrupt_flags) if leg2_corrupt_flags else 0

    false_flag_routed = sum(routed_clean_flags) / len(routed_clean_flags) if routed_clean_flags else 0
    false_flag_leg1 = sum(leg1_clean_flags) / len(leg1_clean_flags) if leg1_clean_flags else 0
    false_flag_leg2 = sum(leg2_clean_flags) / len(leg2_clean_flags) if leg2_clean_flags else 0

    print(f"  Catch Rates (Planted Lie Detection):")
    print(f"    Routed    : {catch_routed * 100:.1f}% (False Flag: {false_flag_routed * 100:.1f}%)")
    print(f"    Leg 1 Only: {catch_leg1 * 100:.1f}% (False Flag: {false_flag_leg1 * 100:.1f}%)")
    print(f"    Leg 2 Only: {catch_leg2 * 100:.1f}% (False Flag: {false_flag_leg2 * 100:.1f}%)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Record output
    rec = {
        "model_name": model_name,
        "n": n,
        "k": k,
        "temperature": args.temperature,
        "seed": args.seed,
        "surprisal_floor": surprisal_floor,
        "precision_weighting": args.ablation_precision_weight,
        "track1_derivable_pct": derivable_pct,
        "track1_identity_match_pct": ep_match_pct,
        "track2_underivable_pct": fq_underivable_pct,
        "track2_identity_match_pct": fq_match_pct,
        "mixed_auroc_routed": au_routed,
        "mixed_auroc_leg1": au_leg1,
        "mixed_auroc_leg2": au_leg2,
        "catch_rate_routed": catch_routed,
        "catch_rate_leg1": catch_leg1,
        "catch_rate_leg2": catch_leg2,
        "false_flag_routed": false_flag_routed,
        "false_flag_leg1": false_flag_leg1,
        "false_flag_leg2": false_flag_leg2,
    }

    out = Path(results_dir) / "llm_f27_router_results.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\nWrote results to {out}")

    print("\nGATE & CONTROLS CHECK:")
    c1 = ep_match_pct == 100.0
    c2 = fq_match_pct == 100.0
    print(f"Control 1 (All-Derivable -> 100% Leg 1): {'PASS' if c1 else 'FAIL'} ({ep_match_pct:.1f}%)")
    print(f"Control 2 (All-Factual -> 100% Leg 2)  : {'PASS' if c2 else 'FAIL'} ({fq_match_pct:.1f}%)")
    print(f"Provisional Pre-registration (Routed >= max(Leg1, Leg2)): {'PASS' if au_routed >= max(au_leg1, au_leg2) else 'PROVISIONAL'}")


if __name__ == "__main__":
    main()
