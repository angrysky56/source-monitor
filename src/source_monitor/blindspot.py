"""
Behavioral instrument: the ghost protocol + gate diagnostics.

Vendored measurement from sps-blindspot (teacher-forced, no generation):
forward the CLEAN and the CORRUPTED trace, read predicted locations at every
emit position after the corruption.

  ghost_follow   fraction of post-corruption predictions equal to the planted
                 token (the model believes its own false emission)
  post_acc       accuracy at post-corruption emit positions (recovery)
  blindspot_idx  final_clean - final_ghost (accuracy lost to one self-error)
  recovery       post-corruption accuracy by distance from the corruption
                 (the sps-blindspot baseline signature: d1 collapses to ~0.28)

New here — gate diagnostics (models with an admission gate):
  gate_auroc     does -gamma rank the corrupted self-position above genuine
                 ones? (pooled over traces; the mechanism check)
  g_corrupt      mean sigmoid(gamma) at corrupted positions (want ~0)
  g_genuine      mean sigmoid(gamma) at genuine positions on the SAME ghosted
                 traces (want ~1)
  g_clean        mean gate on fully clean traces (false-closure check)

Run with injector=inject_ghost for the trained-on corruption type and
injector=inject_mislocation for the HELD-OUT type (the transfer question).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch

from .metrics import auroc
from .model import SMDecoder
from .task import Task, self_positions
from .train import make_tensor, prov_tensor


def make_pairs(
    tasks: list[Task],
    injector: Callable[[Task, random.Random], Task | None],
    rng: random.Random,
) -> list[tuple[Task, Task, int, int]]:
    """(clean, corrupted, step_k, planted_token) for each corruptible task."""
    pairs = []
    for t in tasks:
        g = injector(t, rng)
        if g is None:
            continue
        diff = [i for i in range(len(t.loc_targets))
                if t.loc_targets[i] != g.loc_targets[i]]
        if not diff:
            continue
        k = diff[0]
        pairs.append((t, g, k, g.loc_targets[k]))
    return pairs


@torch.no_grad()
def run_blindspot(
    model: SMDecoder,
    pairs: list[tuple[Task, Task, int, int]],
    device: torch.device,
    batch: int = 256,
) -> dict:
    """Behavioral blind-spot metrics + gate diagnostics on corrupted traces."""
    model.eval()
    emit_pos = list(pairs[0][0].emit_marker_pos)
    self_pos = self_positions(pairs[0][0])
    n_emit = len(emit_pos)

    follow = follow_tot = 0
    post_c = post_tot = 0
    final_clean_c = final_ghost_c = 0
    dist_c: dict[int, int] = {}
    dist_t: dict[int, int] = {}
    gate_scores: list[float] = []   # -gamma (higher = more suspicious)
    gate_labels: list[int] = []     # 1 = corrupted position
    g_corrupt: list[float] = []
    g_genuine: list[float] = []
    g_clean: list[float] = []

    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        clean_tasks = [c for c, _, _, _ in chunk]
        ghost_tasks = [g for _, g, _, _ in chunk]
        clean_x = make_tensor(clean_tasks, device)
        ghost_x = make_tensor(ghost_tasks, device)
        pv = prov_tensor(clean_tasks, device)   # provenance layout is identical
        clean_logits, clean_gate = model(clean_x, pv)
        ghost_logits, ghost_gate = model(ghost_x, pv)
        clean_pred = clean_logits[:, emit_pos, :].argmax(-1)   # (B, n_emit)
        ghost_pred = ghost_logits[:, emit_pos, :].argmax(-1)

        for j, (t, _g, k, gtok) in enumerate(chunk):
            final_clean_c += int(int(clean_pred[j, -1]) == t.answer)
            final_ghost_c += int(int(ghost_pred[j, -1]) == t.answer)
            for e in range(k + 1, n_emit):
                pred = int(ghost_pred[j, e])
                true = t.loc_targets[e]
                post_c += int(pred == true)
                post_tot += 1
                follow += int(pred == gtok and true != gtok)
                follow_tot += 1
                d = e - k
                dist_c[d] = dist_c.get(d, 0) + int(pred == true)
                dist_t[d] = dist_t.get(d, 0) + 1
            if ghost_gate is not None:
                gam = ghost_gate[j, self_pos]              # (n_emit,)
                gv = torch.sigmoid(gam)
                for e in range(n_emit):
                    gate_scores.append(float(-gam[e]))
                    gate_labels.append(int(e == k))
                    (g_corrupt if e == k else g_genuine).append(float(gv[e]))
                g_clean.extend(torch.sigmoid(clean_gate[j, self_pos]).tolist())

    n = len(pairs)
    recovery = {d: dist_c[d] / dist_t[d] for d in sorted(dist_c) if dist_t[d] >= 5}
    model.train()
    out = {
        "n_pairs": n,
        "ghost_follow": follow / follow_tot if follow_tot else float("nan"),
        "post_acc": post_c / post_tot if post_tot else float("nan"),
        "final_clean": final_clean_c / n,
        "final_ghost": final_ghost_c / n,
        "blindspot_idx": (final_clean_c - final_ghost_c) / n,
        "recovery_curve": recovery,
    }
    if gate_scores:
        out["gate_auroc"] = auroc(np.array(gate_scores), np.array(gate_labels))
        out["g_corrupt"] = float(np.mean(g_corrupt))
        out["g_genuine"] = float(np.mean(g_genuine))
        out["g_clean"] = float(np.mean(g_clean))
    return out


def fmt_blindspot(tag: str, r: dict) -> str:
    """One-line summary in the sps-blindspot log style."""
    curve = " ".join(f"d{d}:{v:.2f}" for d, v in list(r["recovery_curve"].items())[:5])
    s = (f"[{tag}] follow={r['ghost_follow']:.3f} post={r['post_acc']:.3f} "
         f"bsi={r['blindspot_idx']:.3f} (clean {r['final_clean']:.3f} -> "
         f"ghost {r['final_ghost']:.3f})  {curve}")
    if "gate_auroc" in r:
        s += (f"\n[{tag}] gate: auroc={r['gate_auroc']:.3f} "
              f"g_corrupt={r['g_corrupt']:.2f} g_genuine={r['g_genuine']:.2f} "
              f"g_clean={r['g_clean']:.2f}")
    return s
