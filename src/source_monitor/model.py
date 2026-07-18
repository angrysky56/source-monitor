"""
Source-monitored decoder: a plain causal LM with two additions.

  1. PROVENANCE EMBEDDING (cfg.provenance): a learned origin embedding
     (external=0 / self=1) added to the token embedding. The origin of a token
     is an architectural given — a generating system always knows which tokens
     it emitted itself — so supplying it is not extra supervision, it is
     stopping the architecture from discarding information it already has.

  2. ADMISSION GATE (cfg.gate != "none"): after `gate_layer` blocks have
     processed the sequence, a scalar head reads each SELF-originated position
     and produces a gate logit γ_i. For all LATER layers, attention logits
     toward that key receive an additive bias logsigmoid(γ_i) ∈ (-inf, 0] — a
     soft, differentiable eviction of the self-emission from the cache.
     External tokens are never gated. Depth-causality makes this realizable in
     sequential generation: when token i is processed, its gate is computed
     once at layer m and stored alongside its KV entries for layers > m.

     The gate head is zero-init with a positive bias, so training starts with
     the gate fully open (exactly a plain decoder) and must LEARN to close it
     on untrustworthy self-emissions — either from task loss alone
     (gate="task", the emergent-source-monitoring arm) or with corruption
     labels (gate="sup", the validated-feasible arm; cf. the sps-blindspot
     confidence-head result, AUROC 0.955-0.968).

This is the "admission" mechanism the sps-blindspot terminal conclusion calls
for: SPS made state authoritative; nothing governed what ENTERS it. In
message-passing terms the standard decoder does backtracking inference — its
own emission returns as fresh evidence; the gate is a learned cavity
correction that can break the echo loop. See SPEC.md for the framing and the
falsifiable predictions.

Dense attention with an explicit additive mask (JVP-friendly, matches the
vendored amplification estimator). RoPE positions. Fine at this scale (3060).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SMConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    rope_base: float = 10000.0
    provenance: bool = False    # add origin embedding to the input
    gate: str = "none"          # "none" | "task" | "sup" | "surprise"
    gate_layer: int = 3         # (task/sup) gate computed after this block
    gate_bias_init: float = 3.0 # sigmoid(3) ~ 0.95: start near-open
    gate_hard: bool = False     # v4 (surprise only): gamma < 0 -> hard evict
                                # (-30), matching the rehearsed dropout
                                # condition exactly; else soft logsigmoid

    @property
    def gated(self) -> bool:
        return self.gate != "none"

    @property
    def learned_gate(self) -> bool:
        """task/sup: a learned scalar head. surprise: self-calibrated surprisal."""
        return self.gate in ("task", "sup")


# ---------------------------------------------------------------------------
# Rotary positional embedding
# ---------------------------------------------------------------------------

def _rope_cache(positions: Tensor, dim: int, base: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables for the given integer positions. positions: (L,) -> (L, dim)."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=positions.device).float() / half))
    ang = positions.float()[:, None] * inv_freq[None, :]  # (L, half)
    ang = torch.cat([ang, ang], dim=-1)                   # (L, dim)
    return ang.cos(), ang.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, L, D). cos/sin: (L, D)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rot * sin[None, None]


def build_causal_mask(seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Positions + plain causal additive mask (0 allowed, -inf blocked)."""
    idx = torch.arange(seq_len, device=device)
    allowed = idx[None, :] <= idx[:, None]
    mask = torch.where(allowed, 0.0, float("-inf"))
    return idx, mask


# ---------------------------------------------------------------------------
# Attention / block — additive mask + optional per-key gate bias
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: SMConfig) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor,
        key_bias: Tensor | None = None,
    ) -> Tensor:
        """key_bias: (B, L) additive per-KEY attention-logit bias (the gate)."""
        b, l, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, self.h, self.dh).transpose(1, 2)  # (B,H,L,dh)
        k = k.view(b, l, self.h, self.dh).transpose(1, 2)
        v = v.view(b, l, self.h, self.dh).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)  # (B,H,L,L)
        att = att + mask[None, None]
        if key_bias is not None:
            att = att + key_bias[:, None, None, :]          # broadcast over queries
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, l, -1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: SMConfig) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor,
        key_bias: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.n1(x), cos, sin, mask, key_bias)
        x = x + self.mlp(self.n2(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SMDecoder(nn.Module):
    """Causal decoder with optional provenance embedding + admission gate."""

    def __init__(self, cfg: SMConfig) -> None:
        super().__init__()
        if cfg.learned_gate and not (1 <= cfg.gate_layer < cfg.n_layers):
            raise ValueError("gate_layer must satisfy 1 <= gate_layer < n_layers "
                             "(the gate must have later layers to act on)")
        if cfg.gate not in ("none", "task", "sup", "surprise"):
            raise ValueError(f"unknown gate mode {cfg.gate!r}")
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.provenance:
            self.prov = nn.Embedding(2, cfg.d_model)
            nn.init.normal_(self.prov.weight, std=0.02)
        if cfg.learned_gate:
            self.gate_norm = nn.LayerNorm(cfg.d_model)
            self.gate_head = nn.Linear(cfg.d_model, 1)
            nn.init.zeros_(self.gate_head.weight)          # start as a no-op:
            nn.init.constant_(self.gate_head.bias, cfg.gate_bias_init)  # open gate
        if cfg.gate == "surprise":
            # γ = softplus(a) * logp(emitted | own marker prediction) + b.
            # At init: genuine (logp≈0) -> γ≈b (open); false emission (logp
            # very negative) -> gate slams shut. No labels, no corruption
            # exposure. softplus SIGN-ANCHORS the calibration (F15b: an idle
            # gate's (a,b) feel ~no gradient and one basin learned an
            # inverted a — surprisal must only ever CLOSE a gate).
            # init 0.5413: softplus(0.5413) ≈ 1.0.
            self.surp_a = nn.Parameter(torch.tensor(0.5413))
            self.surp_b = nn.Parameter(torch.tensor(cfg.gate_bias_init))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def backbone(
        self,
        tokens: Tensor,                 # (B, L) long
        prov: Tensor,                   # (B, L) long in {0,1}; 1 = self-emitted
        perturb: tuple[Tensor, int] | None = None,
        drop: Tensor | None = None,     # (B, L) bool: hard-mask these keys (-30)
    ) -> tuple[Tensor, Tensor | None]:
        """
        Run the network body.

        Returns (post-norm hidden (B,L,D), gate_logit (B,L) or None).
        gate_logit is the raw γ at EVERY position; only SELF positions' values
        are ever applied as biases (external columns get exactly 0 bias).

        perturb=(delta, tau): add `delta` (D,) to the embedding at position tau
        — kept differentiable for the JVP amplification estimator.

        drop: emission-dropout (v3). Training-time hard eviction of sampled
        self-emissions so the model PRACTICES operating across a hole —
        self-supervised, corruption-free rehearsal of the fallback that
        detection-driven eviction needs at test time (FINDINGS F9).
        """
        cfg = self.cfg
        emb = self.tok(tokens)                              # (B,L,D)
        if cfg.provenance:
            emb = emb + self.prov(prov)
        if perturb is not None:
            delta, tau = perturb
            add = torch.zeros_like(emb)
            add[:, tau, :] = delta
            emb = emb + add

        positions, mask = build_causal_mask(tokens.shape[1], tokens.device)
        cos, sin = _rope_cache(positions, cfg.d_model // cfg.n_heads, cfg.rope_base)

        # F12 lesson: trained attention logits can exceed 30, so an additive
        # "eviction" of -30 merely attenuates — the recurrence head shouts
        # through it. True eviction must be -inf-grade (-1e9; not -inf, which
        # can NaN a softmax row).
        EVICT = -1e9
        drop_bias: Tensor | None = None
        if drop is not None:
            drop_bias = torch.where(
                drop, torch.full_like(prov, EVICT, dtype=emb.dtype),
                torch.zeros_like(prov, dtype=emb.dtype))

        def _plus_drop(bias: Tensor | None) -> Tensor | None:
            if drop_bias is None:
                return bias
            return drop_bias if bias is None else bias + drop_bias

        if cfg.gate == "surprise":
            # Pass 1 (ungated): the model's own predictions. The marker state
            # at p-1 never attends to the emission at p (causal), so the
            # prediction is uncontaminated by the token it is checking; at
            # inference this is one streaming pass (the marker state exists
            # before the emission token arrives).
            h1 = emb
            for blk in self.blocks:
                h1 = blk(h1, cos, sin, mask, _plus_drop(None))
            logits1 = self.head(self.norm(h1))
            gate_logit = self._surprise_gamma(tokens, prov, logits1)
            soft = F.logsigmoid(gate_logit)
            if cfg.gate_hard:
                # binary admission: rejected = fully evicted, exactly the
                # condition emission-dropout rehearses (FINDINGS F11/F12)
                soft = torch.where(gate_logit < 0.0,
                                   torch.full_like(soft, EVICT), soft)
            key_bias = torch.where(
                prov.bool(), soft, torch.zeros_like(soft),
            )
            # Pass 2: gated at every layer (the bias is sequence-causal).
            h = emb
            for blk in self.blocks:
                h = blk(h, cos, sin, mask, _plus_drop(key_bias))
            return self.norm(h), gate_logit

        h = emb
        gate_logit: Tensor | None = None
        key_bias: Tensor | None = None
        for li, blk in enumerate(self.blocks, start=1):
            h = blk(h, cos, sin, mask, _plus_drop(key_bias))
            if cfg.learned_gate and li == cfg.gate_layer:
                gate_logit = self.gate_head(self.gate_norm(h)).squeeze(-1)  # (B,L)
                key_bias = torch.where(
                    prov.bool(), F.logsigmoid(gate_logit),
                    torch.zeros_like(gate_logit),
                )
        return self.norm(h), gate_logit

    def _surprise_gamma(self, tokens: Tensor, prov: Tensor, logits1: Tensor) -> Tensor:
        """
        γ (B,L): at each self position p, a * logp(tokens[p] | logits1[p-1]) + b;
        0 elsewhere (external positions are never biased regardless).
        """
        lp_all = F.log_softmax(logits1.float(), dim=-1)
        gamma = torch.zeros(tokens.shape, dtype=lp_all.dtype, device=tokens.device)
        b_idx, p_idx = prov.nonzero(as_tuple=True)
        if b_idx.numel():
            # self positions are emission slots (marker at p-1); p >= 1 by layout
            lp = lp_all[b_idx, p_idx - 1, tokens[b_idx, p_idx]]
            a = F.softplus(self.surp_a)     # sign-anchored (F15b)
            gamma = gamma.index_put((b_idx, p_idx), a * lp + self.surp_b)
        return gamma

    def forward(
        self, tokens: Tensor, prov: Tensor, drop: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """(logits (B,L,V), gate_logit (B,L) or None)."""
        h, gate_logit = self.backbone(tokens, prov, drop=drop)
        return self.head(h), gate_logit
