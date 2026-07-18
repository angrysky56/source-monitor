"""
Minimal self-contained Muon optimizer + parameter routing (vendored from
sps-blindspot; grounded there: Muon >> AdamW on this task, 0.99 vs 0.80 @1000
steps at matched default LRs).

Muon orthogonalizes the momentum update via Newton-Schulz and is applied ONLY
to hidden weight matrices. Everything else — embeddings (token AND provenance),
the LM head, norms, biases, and the gate head — stays on AdamW, routed by ROLE
not by shape (the gate head's (1, d) weight is 2D but is not a y = x @ W body
matrix).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def _newton_schulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Quintic Newton-Schulz iteration -> approximate orthogonalization of a 2D matrix."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum-SGD with orthogonalized updates, for 2D hidden weight matrices only."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        for group in self.param_groups:
            mom, lr, ns = group["momentum"], group["lr"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("mom", torch.zeros_like(p))
                buf.mul_(mom).add_(p.grad)
                update = _newton_schulz5(buf, steps=ns)
                # fan-in scale: keeps update RMS comparable across matrix shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Route parameters: (muon_matrices, adamw_rest). Routed by ROLE:
    Muon gets 2D matrices in the transformer body; AdamW gets tok/prov
    embeddings, LM head, all norms/biases, and the gate head.
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_body_matrix = (
            p.ndim == 2
            and "tok" not in name
            and "prov" not in name
            and "head" not in name      # LM head AND gate_head
            and "gate" not in name
        )
        (muon if is_body_matrix else adamw).append(p)
    return muon, adamw


def build_optimizers(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-3,
    weight_decay: float = 0.0,
    use_muon: bool = True,
) -> list[torch.optim.Optimizer]:
    """
    Two optimizers stepped together. use_muon=False puts everything on AdamW.
    Muon and AdamW LRs are in different units; sweep them on separate grids.
    """
    muon_params, adamw_params = split_params(model)
    if not use_muon:
        return [torch.optim.AdamW(muon_params + adamw_params, lr=adamw_lr,
                                  weight_decay=weight_decay, betas=(0.9, 0.95))]
    opts: list[torch.optim.Optimizer] = [Muon(muon_params, lr=muon_lr)]
    if adamw_params:
        opts.append(torch.optim.AdamW(adamw_params, lr=adamw_lr,
                                      weight_decay=weight_decay, betas=(0.9, 0.95)))
    return opts
