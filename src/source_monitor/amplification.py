"""
Spectral instrument: error-propagation amplification (vendored JVP estimator
from sps-blindspot, extended to the gate's sharpest prediction).

The sibling project measured emission->answer amplification on the CLEAN trace
(baseline: emission/control ratio ~0.54; SPS: 1.50 — state-fidelity flip).
A source-monitoring gate predicts something more specific — SELECTIVE
contraction:

  amp_genuine   perturbation at the genuine emission, clean trace.
                Gate open -> comparable to an ungated baseline.
  amp_corrupt   perturbation at the PLANTED token, corrupted trace.
                Gate closed on a detected-dubious emission -> propagation to
                the final answer should be strongly suppressed vs baseline.
  amp_control   perturbation at a random op token, clean trace (calibrates).

A model that contracts amp_corrupt while keeping amp_genuine is doing exactly
what the cavity/anti-backtracking frame asks: discount self-evidence when (and
only when) it is untrustworthy. fp32 + eager attention (JVP needs
double-backward; bf16 mantissa noise swamps the ~0.01 signal).
"""

from __future__ import annotations

import random

import torch

from .model import SMDecoder
from .task import Task
from .train import make_tensor, prov_tensor


def out_position(task: Task) -> int:
    """Output index whose representation is the final-answer readout."""
    return task.emit_marker_pos[-1]


def amplification(
    model: SMDecoder,
    tokens: torch.Tensor,   # (1, L)
    prov: torch.Tensor,     # (1, L)
    tau: int,               # position to perturb
    out_pos: int,           # position to read
    probes: int = 3,
    iters: int = 3,
    seed: int = 0,
) -> float:
    """Top singular gain of d(hidden[out_pos]) / d(embedding perturbation at tau)."""
    d = model.cfg.d_model
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def f(delta: torch.Tensor) -> torch.Tensor:
        h, _ = model.backbone(tokens, prov, perturb=(delta, tau))
        return h[0, out_pos, :]

    best = 0.0
    for _ in range(probes):
        v = torch.randn(d, generator=gen).to(device=dev, dtype=dtype)
        v = v / v.norm()
        gain = 0.0
        for _ in range(iters):
            _, jv = torch.autograd.functional.jvp(
                f, (torch.zeros(d, device=dev, dtype=dtype),), (v,))
            gain = float(jv.norm())
            if gain == 0.0:
                break
            v = jv / jv.norm()
        best = max(best, gain)
    return best


def measure_arm(
    model: SMDecoder,
    pairs: list[tuple[Task, Task, int, int]],
    device: torch.device,
    n: int,
    rng: random.Random,
) -> dict[str, float]:
    """Mean amplification over `n` pairs: genuine / corrupt / control channels."""
    genuine, corrupt, control = [], [], []
    for (t, g, k, _tok) in pairs[:n]:
        clean_toks = torch.tensor([t.tokens], dtype=torch.long, device=device)
        ghost_toks = torch.tensor([g.tokens], dtype=torch.long, device=device)
        pv = prov_tensor([t], device)
        op = out_position(t)
        tau_emit = t.emit_marker_pos[k] + 1
        genuine.append(amplification(model, clean_toks, pv, tau_emit, op))
        corrupt.append(amplification(model, ghost_toks, pv, tau_emit, op))
        ctrl_tau = rng.choice([p + rng.choice([1, 2]) for p in t.emit_marker_pos[:-1]])
        control.append(amplification(model, clean_toks, pv,
                                     min(ctrl_tau, len(t.tokens) - 2), op))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "amp_genuine": mean(genuine),
        "amp_corrupt": mean(corrupt),
        "amp_control": mean(control),
        "amp_n": float(len(genuine)),
    }


def fmt_amp(tag: str, r: dict[str, float]) -> str:
    ctl = r["amp_control"]
    return (f"[{tag}] amp genuine={r['amp_genuine']:.4f} ({r['amp_genuine']/ctl:.2f}x ctl)  "
            f"corrupt={r['amp_corrupt']:.4f} ({r['amp_corrupt']/ctl:.2f}x ctl)  "
            f"control={ctl:.4f}  (n={int(r['amp_n'])})")
