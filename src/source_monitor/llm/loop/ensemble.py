"""eBP-flavoured ensemble scoring for the detector (F22 / F23).

Motivation (F22c). Pitkow, Ahmadian & Miller (NIPS 2011) show that when an
approximate inference procedure cannot reach a target under ANY parameters, the
remedy is not a better single parameter setting but an average over an ensemble
of perturbed ones — and that the required perturbation covariance is LOW-RANK
(one or two principal components sufficed in their experiments).

Our detector currently reads surprisal off one forward pass at one parameter
setting, i.e. one fixed point, and the F21e floor is calibrated in nats against
that single estimate. This module draws ``k`` members from a low-rank gaussian
parameter ensemble and averages the resulting span scores.

Implementation note — why hooks, not weight edits. A rank-``r`` additive
perturbation ``W -> W + c·B·A`` has exactly the same effect on a linear layer as
adding ``c·(x @ Aᵀ) @ Bᵀ`` to that layer's OUTPUT. Doing it with a forward hook
means: no weight copies (no VRAM doubling), exact restoration on removal (just
drop the hook — no bf16 add/subtract drift accumulating across members), and cost
of two thin matmuls per layer instead of a full re-materialisation.

CAUTION: transformers do not run belief propagation. This is an operational
analogy borrowed from F22, not an implementation of eBP, and nothing here
inherits eBP's convergence guarantees.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from source_monitor.llm.telemetry import retrospective_surprisal

# Qwen3 / Llama-family projection names. These are every linear map in a decoder
# layer; perturbing all of them is the faithful reading of "θ ~ N(θ̄, Σθ)".
DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _target_linears(
    model: nn.Module, patterns: Sequence[str]
) -> list[tuple[str, nn.Linear]]:
    """Every ``nn.Linear`` whose qualified name ends with one of ``patterns``."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(name.endswith(p) for p in patterns)
    ]


def _make_hook(
    weight: Tensor, rank: int, sigma: float, gen: torch.Generator
) -> Any:
    """Build a forward hook applying a rank-``r`` perturbation to the output.

    The factors are drawn ``A ~ N(0, I)`` of shape ``(r, d_in)`` and
    ``B ~ N(0, I)`` of shape ``(d_out, r)``, then scaled so the induced weight
    perturbation has a controlled RELATIVE size::

        E‖B·A‖_F ≈ sqrt(d_out · d_in · r)   (independent standard normals)
        c = sigma · ‖W‖_F / sqrt(d_out · d_in · r)
        ⇒ ‖ΔW‖_F ≈ sigma · ‖W‖_F

    so ``sigma`` reads directly as "fractional perturbation of this layer", which
    is what makes a sweep interpretable and comparable across layers of different
    width.

    Args:
        weight: The layer's weight, used for its shape, dtype, device and norm.
        rank: Perturbation rank ``r`` (the low-rank claim, H-ens-3).
        sigma: Relative perturbation size (0 ⇒ identity, checked by the caller).
        gen: Seeded generator, so a member is reproducible from (seed, index).

    Returns:
        A forward hook of signature ``(module, args, output) -> Tensor``.
    """
    d_out, d_in = weight.shape
    dev = weight.device
    # Draw in float32 for numerical sanity, then cast to the model's dtype.
    a = torch.randn(rank, d_in, generator=gen, device=dev, dtype=torch.float32)
    b = torch.randn(d_out, rank, generator=gen, device=dev, dtype=torch.float32)

    w_norm = float(torch.linalg.norm(weight.detach().float()))
    scale = sigma * w_norm / (d_out * d_in * rank) ** 0.5

    a = (a * scale).to(weight.dtype)
    b = b.to(weight.dtype)

    def hook(_module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        x = args[0]
        return output + (x @ a.T) @ b.T

    return hook


@contextlib.contextmanager
def perturbed(
    model: nn.Module,
    sigma: float,
    rank: int = 8,
    seed: int = 0,
    patterns: Sequence[str] = DEFAULT_TARGETS,
) -> Iterator[int]:
    """Temporarily add a low-rank gaussian perturbation to ``model``'s weights.

    ``sigma == 0`` is a true no-op: no hooks are registered, so scoring inside the
    block is bit-identical to scoring outside it. That is the F23 control.

    Args:
        model: The model to perturb (left unmodified on exit).
        sigma: Relative perturbation size; 0 disables.
        rank: Rank of the perturbation.
        seed: Seed identifying this ensemble member.
        patterns: Linear-layer name suffixes to perturb.

    Yields:
        The number of layers perturbed (0 when ``sigma == 0``).
    """
    if sigma == 0.0:
        yield 0
        return

    targets = _target_linears(model, patterns)
    if not targets:
        raise ValueError(
            f"no nn.Linear matched patterns {tuple(patterns)}; "
            "check the model's module naming"
        )

    handles = []
    try:
        for i, (_name, mod) in enumerate(targets):
            gen = torch.Generator(device=mod.weight.device)
            gen.manual_seed(seed * 100_003 + i)  # distinct stream per layer
            handles.append(
                mod.register_forward_hook(_make_hook(mod.weight, rank, sigma, gen))
            )
        yield len(handles)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def ensemble_span_scores(
    model: Any,
    input_ids: Tensor,
    spans: list,
    k: int = 1,
    sigma: float = 0.0,
    rank: int = 8,
    base_seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Value-only retrospective surprisal averaged over ``k`` ensemble members.

    With ``k == 1`` or ``sigma == 0`` this reduces exactly to the single-pass
    detector of ``loop.monitor.span_scores`` — the two are interchangeable, which
    is what lets F23 attribute any difference to the ensemble and nothing else.

    Args:
        model: A causal LM.
        input_ids: Token ids, shape ``(1, L)``.
        spans: Span annotations from ``tokenize_with_provenance``.
        k: Number of ensemble members (forward passes).
        sigma: Relative perturbation size per member.
        rank: Perturbation rank.
        base_seed: Base seed; member ``j`` uses ``base_seed + j``.

    Returns:
        ``(mean_scores, std_scores)`` — per-span mean over members, and the
        across-member standard deviation (0.0 everywhere when ``sigma == 0``).
        ``std`` is diagnostic: it says how much the estimate actually moved, and
        a sweep where it stays ~0 means the perturbation is too small to matter.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    members: list[list[float]] = []
    for j in range(k):
        with perturbed(model, sigma=sigma, rank=rank, seed=base_seed + j):
            members.append(
                [
                    s.slot_only_neglogp
                    for s in retrospective_surprisal(model, input_ids, spans)
                ]
            )

    n_spans = len(members[0])
    if any(len(m) != n_spans for m in members):
        raise RuntimeError("ensemble members disagree on span count")

    mean: list[float] = []
    std: list[float] = []
    for i in range(n_spans):
        vals = [m[i] for m in members]
        mu = sum(vals) / k
        mean.append(mu)
        std.append((sum((v - mu) ** 2 for v in vals) / k) ** 0.5 if k > 1 else 0.0)
    return mean, std
