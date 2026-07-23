"""F26 viability: does answer-consistency reveal the knowledge boundary that
single-answer surprisal hides?

Setup (factual_qa, grounded=False = pure recall — the F19 ~chance regime):
  known        answerable questions (meta.negation_correct == False)
  unanswerable genuinely unanswerable questions (negation_correct == True)

For each question we compute, over k question paraphrases:
  stability / entropy  of the model's PREFERRED candidate answer (consistency.py)
  min_surprisal        surprisal of the preferred answer on the identity frame
                       (the naive "confidence" a single pass would read)

Pre-registered (F21b discipline — the honest prior is written down):
  H-cons-1  a consistency signal (entropy, or 1-stability, or modal-is-value)
            separates unanswerable from known at AUROC >= .65
  H-cons-2  min_surprisal does NOT (AUROC <= .60) — single-answer confidence is
            blind to the knowledge boundary, so consistency ADDS a leg surprisal
            can't provide.
HONEST PRIOR: Qwen3-1.7B may be well enough calibrated to STABLY prefer the
"no reliable record" abstention on unanswerable questions. If so the separator is
modal_is_value (known->value, unanswerable->negation), NOT instability — which is
still a working factual leg, just via preferred-answer identity rather than
dispersion. Report all three; let the data pick.

Run:  python -m source_monitor.llm.loop.f26_consistency --quick
      python -m source_monitor.llm.loop.f26_consistency
GPU HEAT: k*n_candidates forward passes per question. Fans on; watch nvidia-smi.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from source_monitor.llm.cache import load_model
from source_monitor.llm.loop.config import Phase3Config
from source_monitor.llm.loop.consistency import answer_stability
from source_monitor.llm.loop.f22_ensemble import auroc
from source_monitor.llm.ood import factual_qa
from source_monitor.llm.ood.base import make_variant, raw_claim_score


@torch.no_grad()
def _min_surprisal(model, tok, trace, device: str) -> float:
    """Value-slot surprisal of the model's best VALUE guess (identity frame).

    Value candidates only and slot-only, matching consistency.preferred_candidate,
    so the baseline is the confidence a single pass would read for the model's
    best answer — the thing consistency is meant to beat.
    """
    surfaces = trace.claim.candidate_surfaces or []
    idxs = [i for i, s in enumerate(surfaces) if s == "value"] or list(
        range(len(trace.claim.candidate_contents))
    )
    return min(
        raw_claim_score(model, tok, make_variant(trace, i), device).value_only_neglogp
        for i in idxs
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="F26 consistency viability")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = Phase3Config()
    model_name = args.model or cfg.model_name
    n = args.n or (16 if args.quick else 60)
    k = 3 if args.quick else args.k
    results_dir = args.results_dir or cfg.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"model={model_name} n={n} k={k} seed={args.seed}")
    model, tok, _ = load_model(model_name, device=cfg.device, dtype=cfg.dtype,
                               enable_thinking=False)
    model.eval()

    traces = factual_qa.generate(args.seed, n)
    known = [t for t in traces if not t.meta.get("negation_correct")]
    unans = [t for t in traces if t.meta.get("negation_correct")]
    print(f"known={len(known)} unanswerable={len(unans)}", flush=True)

    def measure(ts):
        rows = []
        for t in ts:
            st = answer_stability(model, tok, t, cfg.device, k=k)
            st["min_surprisal"] = _min_surprisal(model, tok, t, cfg.device)
            rows.append(st)
        return rows

    t0 = time.time()
    k_rows = measure(known)
    u_rows = measure(unans)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    def col(rows, key):
        return [float(r[key]) for r in rows]

    # AUROC: positive class = unanswerable (the "should-flag / model-doesn't-know").
    au_entropy = auroc(col(u_rows, "entropy"), col(k_rows, "entropy"))
    au_instab = auroc([1 - s for s in col(u_rows, "stability")],
                      [1 - s for s in col(k_rows, "stability")])
    # modal_is_value: known should be value, unanswerable should be negation -> use
    # value-ness as a "known" signal, so flag = NOT value.
    au_notvalue = auroc([0.0 if r["modal_is_value"] else 1.0 for r in u_rows],
                        [0.0 if r["modal_is_value"] else 1.0 for r in k_rows])
    au_surp = auroc(col(u_rows, "min_surprisal"), col(k_rows, "min_surprisal"))

    def mean(rows, key):
        return float(np.mean(col(rows, key))) if rows else float("nan")

    def frac_value(rows):
        return float(np.mean([r["modal_is_value"] for r in rows])) if rows else float("nan")

    rec = {"model_name": model_name, "n": n, "k": k, "seed": args.seed,
           "n_known": len(known), "n_unans": len(unans),
           "auroc_entropy": au_entropy, "auroc_instability": au_instab,
           "auroc_not_value": au_notvalue, "auroc_min_surprisal": au_surp,
           "known_stability": mean(k_rows, "stability"),
           "unans_stability": mean(u_rows, "stability"),
           "known_frac_value": frac_value(k_rows),
           "unans_frac_value": frac_value(u_rows),
           "known_modal_correct": mean(k_rows, "modal_correct"),
           "unans_modal_correct": mean(u_rows, "modal_correct")}
    out = Path(results_dir) / "llm_f26_consistency_results.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 66)
    print(f"{'group':<14}{'stability':<12}{'frac_value':<12}{'modal_correct'}")
    print(f"{'known':<14}{rec['known_stability']:<12.3f}{rec['known_frac_value']:<12.3f}"
          f"{rec['known_modal_correct']:.3f}")
    print(f"{'unanswerable':<14}{rec['unans_stability']:<12.3f}{rec['unans_frac_value']:<12.3f}"
          f"{rec['unans_modal_correct']:.3f}")
    print("\nAUROC (separating unanswerable from known):")
    print(f"  entropy           {au_entropy:.3f}")
    print(f"  instability(1-st) {au_instab:.3f}")
    print(f"  modal-not-value   {au_notvalue:.3f}")
    print(f"  min_surprisal     {au_surp:.3f}   (baseline; single-answer confidence)")

    best_cons = max(au_entropy, au_instab, au_notvalue)
    print("\nGATE")
    print(f"H-cons-1  best consistency signal >= .65: {best_cons:.3f}  "
          f"[{'PASS' if best_cons >= 0.65 else 'FAIL'}]")
    print(f"H-cons-2  min_surprisal <= .60 (blind): {au_surp:.3f}  "
          f"[{'PASS' if au_surp <= 0.60 else 'FAIL'}]")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
