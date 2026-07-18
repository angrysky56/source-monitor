"""
Orchestrator: the five-arm matched comparison, single process, durable output.

(Per the sps-blindspot infra note: long runs die with the shell session, so
everything loops inside ONE invocation and appends to results/results.jsonl
as each arm finishes — a crash loses at most the current arm.)

Arms (all identical backbone, optimizer, data, steps; see SPEC.md §5):
  base-clean  no prov, no gate, ghost_frac 0.0   — replicates the original
              blind spot (expect d1 ~0.28, bsi ~0.11: the anchor)
  base-mix    no prov, no gate, ghost_frac F     — THE control: what does data
              exposure alone buy?
  prov-mix    provenance embedding only          — is knowing "this is mine"
              enough for the model to discount it?
  gate-task   prov + gate, task loss only        — EMERGENT source monitoring
  gate-sup    prov + gate, corruption-labeled    — validated-feasible upper arm

Per arm: clean competence, ghost protocol (trained-on corruption), mislocation
protocol (HELD-OUT corruption — the transfer question), gate diagnostics, and
optionally the JVP amplification triplet (--amp-n 0 to skip).

    python -m source_monitor.experiments --seeds 0,1,2 --steps 1500
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import time

import torch

from .amplification import fmt_amp, measure_arm
from .blindspot import fmt_blindspot, make_pairs, run_blindspot
from .task import (
    generate_dataset,
    inject_ghost,
    inject_mislocation,
    inject_phantom_removal,
)
from .train import add_common_args, evaluate, make_config, train_inline

ARM_DEFS: dict[str, dict] = {
    "base-clean": dict(provenance=False, gate="none", ghost_frac=0.0),
    "base-mix":   dict(provenance=False, gate="none", ghost_frac=None),
    "prov-mix":   dict(provenance=True,  gate="none", ghost_frac=None),
    "gate-task":  dict(provenance=True,  gate="task", ghost_frac=None),
    "gate-sup":   dict(provenance=True,  gate="sup",  ghost_frac=None),
    # v2 (post five-arm run, see FINDINGS): label-free surprisal gate.
    # surp-clean NEVER sees a corruption in training — the pure transfer arm.
    "surp-clean": dict(provenance=True,  gate="surprise", ghost_frac=0.0),
    "surp-mix":   dict(provenance=True,  gate="surprise", ghost_frac=None),
    # v3 (post v2 run, FINDINGS F8/F9): detection transferred, eviction did
    # not repair. Emission dropout rehearses operating-across-a-hole,
    # corruption-free. surp-drop = detection + practiced fallback (P8);
    # base-drop = the dropout-only control.
    "surp-drop":  dict(provenance=True,  gate="surprise", ghost_frac=0.0,
                       emission_dropout=0.3),
    "base-drop":  dict(provenance=False, gate="none", ghost_frac=0.0,
                       emission_dropout=0.3),
    # v4 (post v3, FINDINGS F11): soft eviction != the rehearsed fully-absent
    # condition. Hard admission: gamma < 0 -> -30 (binary evict). Completes
    # the 2x2 {rehearsal} x {hard eviction} at fixed detection.
    "surp-clean-hard": dict(provenance=True, gate="surprise", ghost_frac=0.0,
                            gate_hard=True),
    "surp-drop-hard":  dict(provenance=True, gate="surprise", ghost_frac=0.0,
                            emission_dropout=0.3, gate_hard=True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--arms", default="base-clean,base-mix,prov-mix,gate-task,gate-sup")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--amp-n", type=int, default=60, help="pairs for the JVP estimator (0 = skip)")
    ap.add_argument("--eval-pairs", type=int, default=1200)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        print("NOTE: GPU run — the 3060's fan is broken; start the floor fans "
              "and watch nvidia-smi.", flush=True)

    tkw = dict(n_ops=args.n_ops, n_objects=args.n_objects,
               n_containers=args.n_containers)
    val = generate_dataset(args.val_tasks, base_seed=999, **tkw)
    raw = generate_dataset(6000, base_seed=555, balance_absent=False, **tkw)
    ghost_pairs = make_pairs([t for t in raw if t.query_obj_removed],
                             inject_ghost, random.Random(123))[: args.eval_pairs]
    misloc_pairs = make_pairs(raw, inject_mislocation,
                              random.Random(321))[: args.eval_pairs]
    phantom_pairs = make_pairs(raw, inject_phantom_removal,
                               random.Random(213))[: args.eval_pairs]
    print(f"eval pairs: ghost={len(ghost_pairs)} misloc={len(misloc_pairs)} "
          f"phantom={len(phantom_pairs)}", flush=True)

    outdir = pathlib.Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "results.jsonl"

    for seed in [int(s) for s in args.seeds.split(",")]:
        for arm in args.arms.split(","):
            arm = arm.strip()
            d = ARM_DEFS[arm]
            ns = copy.deepcopy(args)
            ns.seed = seed
            ns.ghost_frac = d["ghost_frac"] if d["ghost_frac"] is not None else args.ghost_frac
            ns.emission_dropout = d.get("emission_dropout", args.emission_dropout)
            cfg = make_config(ns, provenance=d["provenance"], gate=d["gate"],
                              gate_hard=d.get("gate_hard", False))
            tag = f"{arm}/s{seed}"
            t0 = time.time()
            model = train_inline(cfg, ns, dev).float()   # fp32 for the JVP
            comp = evaluate(model, val, dev)
            gr = run_blindspot(model, ghost_pairs, dev)
            mr = run_blindspot(model, misloc_pairs, dev)
            pr = run_blindspot(model, phantom_pairs, dev)
            amp = (measure_arm(model, ghost_pairs, dev, args.amp_n, random.Random(7))
                   if args.amp_n > 0 else {})
            wall = time.time() - t0

            print(f"\n[{tag}] competence acc={comp['acc']:.3f} "
                  f"perstep={comp['acc_perstep']:.3f}  ({wall:.0f}s)", flush=True)
            print(fmt_blindspot(f"{tag}/ghost  ", gr), flush=True)
            print(fmt_blindspot(f"{tag}/misloc ", mr), flush=True)
            print(fmt_blindspot(f"{tag}/phantom", pr), flush=True)
            if amp:
                print(fmt_amp(f"{tag}/amp    ", amp), flush=True)

            rec = {
                "arm": arm, "seed": seed, "ghost_frac": ns.ghost_frac,
                "emission_dropout": ns.emission_dropout,
                "steps": ns.steps, "d_model": ns.d_model, "n_layers": ns.n_layers,
                "wall_s": round(wall, 1), "competence": comp,
                "ghost": {k: v for k, v in gr.items() if k != "recovery_curve"},
                "ghost_recovery": {str(k): v for k, v in gr["recovery_curve"].items()},
                "misloc": {k: v for k, v in mr.items() if k != "recovery_curve"},
                "misloc_recovery": {str(k): v for k, v in mr["recovery_curve"].items()},
                "phantom": {k: v for k, v in pr.items() if k != "recovery_curve"},
                "phantom_recovery": {str(k): v for k, v in pr["recovery_curve"].items()},
                **amp,
            }
            with outfile.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\nresults appended -> {outfile}", flush=True)
    print("Predictions (SPEC.md §4): base-clean replicates d1~0.28/bsi~0.11; "
          "gate arms should hold d1 high, close on ghosts (gate_auroc), "
          "transfer to misloc, and contract amp_corrupt but not amp_genuine.",
          flush=True)


if __name__ == "__main__":
    main()
