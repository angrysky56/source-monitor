"""
F12 diagnostic: why doesn't hard eviction repair?

The paradox: gate-hard evicts the ghost key (-30, every layer of the output
pass) and detection is ~perfect, yet d1 sits at the blind-spot floor — while
the same model was TRAINED (emission dropout) to predict correctly across
hard-masked emissions and does so at ceiling.

Three-way forward comparison on ghosted traces, same trained model
(surp-drop-hard config, seed 0):

  GATE   forward(x_ghost)              — gate-hard does the evicting
  DROP   forward(x_ghost, drop=ghost)  — the exact training-time mask,
                                          forced, bypassing gate judgment
  CLEAN  forward(x_clean)              — reference

Measured at the d1 marker (first emit after the ghost):
  d1 accuracy under each condition, and cosine similarity of the final
  hidden state at that marker: cos(GATE, DROP) tells us whether the gate's
  eviction is mechanically equivalent to the rehearsed mask; d1(DROP) tells
  us whether even the rehearsed mask repairs.

Readings:
  d1(DROP) high, d1(GATE) low  -> gate eviction mechanically != drop (bug or
                                   leak in the gate path). Fix the gate.
  d1(DROP) low too             -> masking truly does not repair: the model
                                   does not deploy its trained hole-skill on
                                   corrupted traces. Deep finding, gate fine.
  cos(GATE,DROP) ~ 1           -> paths equivalent; behavior must match.
"""

from __future__ import annotations

import argparse
import random

import torch

from .blindspot import make_pairs
from .model import SMDecoder
from .task import generate_dataset, inject_ghost
from .train import add_common_args, make_config, make_tensor, prov_tensor, train_inline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n-pairs", type=int, default=300)
    args = ap.parse_args()
    args.ghost_frac = 0.0
    args.emission_dropout = 0.3
    dev = torch.device(args.device)
    if dev.type == "cuda":
        print("NOTE: GPU run — fans on, watch nvidia-smi.", flush=True)

    cfg = make_config(args, provenance=True, gate="surprise", gate_hard=True)
    model = train_inline(cfg, args, dev).float().eval()
    torch.set_grad_enabled(False)   # measurement from here on
    print("trained surp-drop-hard config (seed %d)" % args.seed, flush=True)

    tkw = dict(n_ops=args.n_ops, n_objects=args.n_objects,
               n_containers=args.n_containers)
    raw = generate_dataset(6000, base_seed=555, balance_absent=False, **tkw)
    pairs = make_pairs([t for t in raw if t.query_obj_removed],
                       inject_ghost, random.Random(123))
    n_emit = len(pairs[0][0].emit_marker_pos)
    pairs = [pr for pr in pairs if pr[2] < n_emit - 1][: args.n_pairs]
    print(f"pairs with a d1 position: {len(pairs)}", flush=True)

    acc = {"GATE": 0, "DROP": 0, "CLEAN": 0, "DROPC": 0}
    cos_gd, cos_gc = [], []
    B = 128
    for i in range(0, len(pairs), B):
        chunk = pairs[i : i + B]
        clean_x = make_tensor([c for c, _, _, _ in chunk], dev)
        ghost_x = make_tensor([g for _, g, _, _ in chunk], dev)
        pv = prov_tensor([c for c, _, _, _ in chunk], dev)
        drop = torch.zeros_like(ghost_x, dtype=torch.bool)
        d1_marker = torch.tensor(
            [t.emit_marker_pos[k + 1] for (t, _, k, _) in chunk], device=dev)
        for j, (t, _, k, _) in enumerate(chunk):
            drop[j, t.emit_marker_pos[k] + 1] = True

        h_gate, _ = model.backbone(ghost_x, pv)
        h_drop, _ = model.backbone(ghost_x, pv, drop=drop)
        h_clean, _ = model.backbone(clean_x, pv)
        # the missing certification: was post-hole re-derivation ever learned?
        h_dropc, _ = model.backbone(clean_x, pv, drop=drop)
        rows = torch.arange(len(chunk), device=dev)
        for name, h in (("GATE", h_gate), ("DROP", h_drop), ("CLEAN", h_clean),
                        ("DROPC", h_dropc)):
            logits = model.head(h[rows, d1_marker])
            preds = logits.argmax(-1)
            for j, (t, _, k, _) in enumerate(chunk):
                acc[name] += int(int(preds[j]) == t.loc_targets[k + 1])
        g = h_gate[rows, d1_marker]
        d = h_drop[rows, d1_marker]
        c = h_clean[rows, d1_marker]
        cos_gd += torch.cosine_similarity(g, d, dim=-1).tolist()
        cos_gc += torch.cosine_similarity(g, c, dim=-1).tolist()

    n = len(pairs)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"\nd1 accuracy:  GATE {acc['GATE']/n:.3f}   DROP {acc['DROP']/n:.3f}   "
          f"CLEAN {acc['CLEAN']/n:.3f}   DROP-on-CLEAN {acc['DROPC']/n:.3f}")
    print("(DROP-on-CLEAN low => dropout training never taught re-derivation; "
          "the rehearsal itself failed, not its transfer)")
    print(f"cos(GATE,DROP) at d1 marker: {mean(cos_gd):.4f}   "
          f"cos(GATE,CLEAN): {mean(cos_gc):.4f}")
    print("\nreadings: DROP high & GATE low -> gate path leaks (fix gate); "
          "DROP low -> masking does not repair (deep finding); "
          "cos(GATE,DROP)~1 -> paths mechanically equivalent.")


if __name__ == "__main__":
    main()
