"""F26c viability: does SAMPLED self-consistency separate known from confabulated
where teacher-forced preference could not (F26b, AUROC .500)?

The discrete-symmetry instrument: draw k independent generations per question and
measure whether the answer holds its shape. Known facts should be generated
consistently (low distinct_ratio); unanswerable questions should either vary
(confabulation, high distinct_ratio) or abstain (high hedge_rate) — both are
"not a confident stable value", i.e. flaggable.

Pre-registered:
  H-samp-1  best of {distinct_ratio, hedge_rate, 1-agreement} separates
            unanswerable from known at AUROC >= .65 (teacher forcing gave .500)
  H-samp-2  the model actually answers known facts (known correct_rate > .5),
            else the task is too hard to interpret the consistency signal.

Run:  python -m source_monitor.llm.loop.f26_sample --quick
      python -m source_monitor.llm.loop.f26_sample
GPU HEAT: k generations per question. Fans on; watch nvidia-smi.
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
from source_monitor.llm.loop.consistency import sampled_consistency
from source_monitor.llm.loop.f22_ensemble import auroc
from source_monitor.llm.ood import factual_qa


def main() -> None:
    ap = argparse.ArgumentParser(description="F26c sampled-consistency viability")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = Phase3Config()
    model_name = args.model or cfg.model_name
    n = args.n or (16 if args.quick else 60)
    k = args.k or (4 if args.quick else 6)
    results_dir = args.results_dir or cfg.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"model={model_name} n={n} k={k} temp={args.temperature} seed={args.seed}")
    model, tok, _ = load_model(model_name, device=cfg.device, dtype=cfg.dtype,
                               enable_thinking=False)
    model.eval()

    traces = factual_qa.generate(args.seed, n)
    known = [t for t in traces if not t.meta.get("negation_correct")]
    unans = [t for t in traces if t.meta.get("negation_correct")]
    print(f"known={len(known)} unanswerable={len(unans)}", flush=True)

    def measure(ts, s0):
        return [sampled_consistency(model, tok, t, cfg.device, k=k,
                                    temperature=args.temperature, seed=s0 + i)
                for i, t in enumerate(ts)]

    t0 = time.time()
    k_rows = measure(known, 0)
    u_rows = measure(unans, 5000)
    print(f"sampled in {time.time() - t0:.0f}s", flush=True)

    # peek at a few raw answers so the mechanism is auditable, not a black box
    for label, rows, ts in (("known", k_rows, known), ("unans", u_rows, unans)):
        if rows:
            print(f"  [{label}] q={ts[0].turns[ts[0].claim.turn_index - 1].content[:40]!r} "
                  f"answers={[a[:18] for a in rows[0]['answers']]}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    def col(rows, key):
        return [float(r[key]) for r in rows]

    def mean(rows, key):
        vals = [v for v in col(rows, key) if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    au_distinct = auroc(col(u_rows, "distinct_ratio"), col(k_rows, "distinct_ratio"))
    au_hedge = auroc(col(u_rows, "hedge_rate"), col(k_rows, "hedge_rate"))
    au_disagree = auroc([1 - a for a in col(u_rows, "agreement")],
                        [1 - a for a in col(k_rows, "agreement")])
    best = max(au_distinct, au_hedge, au_disagree)

    rec = {"model_name": model_name, "n": n, "k": k, "temperature": args.temperature,
           "seed": args.seed, "n_known": len(known), "n_unans": len(unans),
           "auroc_distinct": au_distinct, "auroc_hedge": au_hedge,
           "auroc_disagree": au_disagree,
           "known_distinct": mean(k_rows, "distinct_ratio"),
           "unans_distinct": mean(u_rows, "distinct_ratio"),
           "known_hedge": mean(k_rows, "hedge_rate"),
           "unans_hedge": mean(u_rows, "hedge_rate"),
           "known_correct": mean(k_rows, "correct_rate")}
    out = Path(results_dir) / "llm_f26_sample_results.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 66)
    print(f"{'group':<14}{'distinct':<11}{'hedge':<10}{'correct'}")
    print(f"{'known':<14}{rec['known_distinct']:<11.3f}{rec['known_hedge']:<10.3f}"
          f"{rec['known_correct']:.3f}")
    print(f"{'unanswerable':<14}{rec['unans_distinct']:<11.3f}{rec['unans_hedge']:<10.3f}"
          f"{'--'}")
    print("\nAUROC (separating unanswerable from known):")
    print(f"  distinct_ratio  {au_distinct:.3f}")
    print(f"  hedge_rate      {au_hedge:.3f}")
    print(f"  disagreement    {au_disagree:.3f}")

    print("\nGATE")
    print(f"H-samp-1  best signal >= .65: {best:.3f}  [{'PASS' if best >= 0.65 else 'FAIL'}]")
    print(f"H-samp-2  known answered (correct_rate > .5): {rec['known_correct']:.3f}  "
          f"[{'PASS' if rec['known_correct'] > 0.5 else 'FAIL'}]")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
