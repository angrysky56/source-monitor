"""Phase 1 runner: OOD transfer of the retrospective self-consistency signal.

For each domain x model x seed, scores genuine + value-corrupted + negation-
corrupted claims with raw (and contrastive where candidates are enumerable),
computes pooled and matched-surface AUROC per aggregation, paired deltas, and a
per-domain calibration affine (Platt) with ECE. Appends flat records to
results/llm_phase1_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from source_monitor.metrics import auroc
from source_monitor.llm.cache import load_model
from source_monitor.llm.ood import (
    arithmetic,
    code_trace,
    entity_prose,
    factual_grounded,
    factual_qa,
)
from source_monitor.llm.ood.base import (
    ClaimScore,
    OODTrace,
    contrastive_claim_score,
    corrupt_to_negation,
    corrupt_to_value,
    raw_claim_score,
)

# domain name -> (module, enumerable_candidates?)
DOMAINS: dict[str, tuple[Any, bool]] = {
    "entity_prose": (entity_prose, True),
    "arithmetic": (arithmetic, True),
    "factual_qa": (factual_qa, True),
    "factual_grounded": (factual_grounded, True),
    "code_trace": (code_trace, False),
}
AGGS = ("mean", "max", "value")
_ATTR = {"mean": "mean_neglogp", "max": "max_neglogp", "value": "value_only_neglogp"}


def _val(s: ClaimScore, agg: str) -> float:
    return float(getattr(s, _ATTR[agg]))


def _auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return auroc(pos + neg, [1] * len(pos) + [0] * len(neg))


def _platt(scores: list[float], labels: list[int], iters: int = 500, lr: float = 0.1) -> tuple[float, float, float]:
    """1-D logistic (Platt) fit score->P(corrupt); returns (slope, bias, ECE)."""
    if len(set(labels)) < 2:
        return 0.0, 0.0, float("nan")
    x = np.array(scores, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        a -= lr * float(np.mean((p - y) * x))
        b -= lr * float(np.mean(p - y))
    p = 1.0 / (1.0 + np.exp(-(a * x + b)))
    # ECE over 10 bins
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() > 0:
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(a), float(b), float(ece)


def run_domain(model, tokenizer, name: str, seed: int, n: int, device: str) -> dict:
    mod, enumerable = DOMAINS[name]
    traces = mod.generate(seed, n)
    rng = random.Random(seed + 1)

    # scorer -> agg -> population -> list[float]
    pops: dict[str, dict[str, dict[str, list[float]]]] = {
        sc: {a: {k: [] for k in ("gen_value", "gen_neg", "cor_value", "cor_neg")} for a in AGGS}
        for sc in ("raw", "contrastive")
    }

    def score(tr: OODTrace):
        out = {"raw": raw_claim_score(model, tokenizer, tr, device)}
        out["contrastive"] = (
            contrastive_claim_score(model, tokenizer, tr, device) if enumerable else None
        )
        return out

    for clean in traces:
        surf = clean.claim.surface_type
        s = score(clean)
        for sc in ("raw", "contrastive"):
            if s[sc] is None:
                continue
            for a in AGGS:
                pops[sc][a]["gen_value" if surf == "value" else "gen_neg"].append(_val(s[sc], a))

        cv = corrupt_to_value(clean, rng) if enumerable else mod.corrupt_value(clean, rng)
        if cv is not None:
            s = score(cv)
            for sc in ("raw", "contrastive"):
                if s[sc] is None:
                    continue
                for a in AGGS:
                    pops[sc][a]["cor_value"].append(_val(s[sc], a))

        cn = corrupt_to_negation(clean) if enumerable else mod.corrupt_negation(clean)
        if cn is not None:
            s = score(cn)
            for sc in ("raw", "contrastive"):
                if s[sc] is None:
                    continue
                for a in AGGS:
                    pops[sc][a]["cor_neg"].append(_val(s[sc], a))

    # metrics
    result: dict[str, Any] = {"domain": name, "seed": seed, "n_traces": n, "scorers": {}}
    for sc in ("raw", "contrastive"):
        if sc == "contrastive" and not enumerable:
            continue
        agg_out = {}
        for a in AGGS:
            p = pops[sc][a]
            gen_all = p["gen_value"] + p["gen_neg"]
            cor_all = p["cor_value"] + p["cor_neg"]
            slope, bias, ece = _platt(cor_all + gen_all, [1] * len(cor_all) + [0] * len(gen_all))
            agg_out[a] = {
                "matched_value_auroc": _auroc(p["cor_value"], p["gen_value"]),
                "matched_negation_auroc": _auroc(p["cor_neg"], p["gen_neg"]),
                "pooled_auroc": _auroc(cor_all, gen_all),
                "n_gen_value": len(p["gen_value"]), "n_gen_neg": len(p["gen_neg"]),
                "n_cor_value": len(p["cor_value"]), "n_cor_neg": len(p["cor_neg"]),
                "calib_slope": slope, "calib_bias": bias, "calib_ece": ece,
            }
        result["scorers"][sc] = agg_out
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 OOD transfer runner")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    ap.add_argument("--n-traces", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2024])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    out_file = Path("results") / "llm_phase1_results.jsonl"

    for model_name in args.models:
        print(f"\n=== Loading {model_name} ===")
        model, tok, meta = load_model(model_name, device=args.device, dtype=args.dtype,
                                      enable_thinking=False)
        for name in args.domains:
            for seed in args.seeds:
                t0 = time.time()
                rec = run_domain(model, tok, name, seed, args.n_traces, args.device)
                rec["model_name"] = model_name
                rec["meta"] = meta.to_dict()
                rec["wall_seconds"] = time.time() - t0
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                r = rec["scorers"]
                best = r.get("contrastive", r["raw"])["mean"]
                print(f"  {name:<13} seed{seed} {rec['wall_seconds']:.0f}s | "
                      f"value AUROC(mean,best)={best['matched_value_auroc']:.3f} "
                      f"neg={best['matched_negation_auroc']:.3f}")
        del model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
