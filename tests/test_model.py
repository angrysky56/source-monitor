"""Model correctness: causality, gate targeting, gate liveness, loss paths, JVP.

All CPU, tiny config — these are the guardrails (a subtly wrong mask or gate
silently breaks the whole experiment; cf. sps-blindspot milestone 1).
"""

from __future__ import annotations

import argparse
import random

import torch

from source_monitor.amplification import amplification, out_position
from source_monitor.model import SMConfig, SMDecoder
from source_monitor.task import VOCAB_SIZE, generate_dataset
from source_monitor.train import Bundle, batch_loss, build_bundle, evaluate, make_tensor, prov_tensor

TKW = dict(n_ops=4, n_objects=3, n_containers=2)


def tiny_cfg(**kw) -> SMConfig:
    base = dict(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2, n_layers=3,
                d_ff=64, gate_layer=2)
    base.update(kw)
    return SMConfig(**base)


def test_causality_plain():
    torch.manual_seed(0)
    m = SMDecoder(tiny_cfg())
    x = torch.randint(0, VOCAB_SIZE, (1, 20))
    pv = torch.zeros(1, 20, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (1, 20))
    j = 11
    x2 = x.clone()
    x2[:, j + 1 :] = y[:, j + 1 :]
    with torch.no_grad():
        a, _ = m(x, pv)
        b, _ = m(x2, pv)
    assert torch.allclose(a[:, : j + 1], b[:, : j + 1], atol=1e-5), \
        "future tokens leaked into past logits"
    assert not torch.allclose(a[:, j + 1 :], b[:, j + 1 :], atol=1e-5)


def test_gate_targets_only_self_columns():
    """With all-external provenance the gate must be a bit-exact no-op,
    whatever the gate head says."""
    torch.manual_seed(1)
    m = SMDecoder(tiny_cfg(gate="task"))
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    ext = torch.zeros(2, 16, dtype=torch.long)
    with torch.no_grad():
        a, _ = m(x, ext)
        m.gate_head.bias.fill_(-30.0)   # slam the gate shut
        b, _ = m(x, ext)
    assert torch.equal(a, b), "gate leaked onto external tokens"


def test_gate_liveness_and_query_causality():
    """A closed gate on a self position must change ONLY queries >= that
    position (earlier queries cannot attend to it)."""
    torch.manual_seed(2)
    m = SMDecoder(tiny_cfg(gate="task"))
    x = torch.randint(0, VOCAB_SIZE, (1, 16))
    p = 9
    pv = torch.zeros(1, 16, dtype=torch.long)
    pv[0, p] = 1
    with torch.no_grad():
        a, ga = m(x, pv)                 # open (init bias +3)
        m.gate_head.bias.fill_(-30.0)
        b, gb = m(x, pv)                 # closed
    assert ga is not None and gb is not None
    assert torch.allclose(a[:, :p], b[:, :p], atol=1e-5), \
        "closed gate changed queries before the gated key"
    assert not torch.allclose(a[:, p:], b[:, p:], atol=1e-5), \
        "gate had no effect — not live"


def test_provenance_embedding_flag():
    torch.manual_seed(3)
    x = torch.randint(0, VOCAB_SIZE, (1, 12))
    pv = torch.zeros(1, 12, dtype=torch.long)
    pv[0, 5] = 1
    m_off = SMDecoder(tiny_cfg(provenance=False))
    with torch.no_grad():
        a, _ = m_off(x, pv)
        b, _ = m_off(x, torch.zeros_like(pv))
    assert torch.equal(a, b), "provenance must be inert when disabled"
    m_on = SMDecoder(tiny_cfg(provenance=True))
    with torch.no_grad():
        c, _ = m_on(x, pv)
        d, _ = m_on(x, torch.zeros_like(pv))
    assert not torch.allclose(c, d, atol=1e-5), "provenance embedding inert when enabled"


def _bundle(n: int = 24) -> tuple[Bundle, list]:
    tasks = generate_dataset(n, base_seed=1, **TKW)
    return build_bundle(tasks, torch.device("cpu"), seed=0), tasks


def test_loss_paths_finite():
    torch.manual_seed(4)
    bundle, _ = _bundle()
    idx = torch.arange(8)
    for gate in ("none", "task", "sup"):
        m = SMDecoder(tiny_cfg(gate=gate, provenance=True))
        loss = batch_loss(m, bundle, idx, ghost_frac=0.5, gate_loss_w=0.5)
        assert torch.isfinite(loss), f"non-finite loss for gate={gate}"
        loss.backward()
        if gate == "sup":
            gw = m.gate_head.weight.grad
            assert gw is not None and torch.isfinite(gw).all()


def test_evaluate_smoke():
    torch.manual_seed(5)
    _, tasks = _bundle()
    m = SMDecoder(tiny_cfg(provenance=True, gate="sup"))
    r = evaluate(m, tasks, torch.device("cpu"))
    assert 0.0 <= r["acc"] <= 1.0 and 0.0 <= r["acc_perstep"] <= 1.0


def test_jvp_through_gate():
    """The amplification estimator must double-backward through the gate path."""
    torch.manual_seed(6)
    _, tasks = _bundle()
    t = tasks[0]
    m = SMDecoder(tiny_cfg(provenance=True, gate="task")).float()
    toks = torch.tensor([t.tokens], dtype=torch.long)
    pv = prov_tensor([t], torch.device("cpu"))
    tau = t.emit_marker_pos[0] + 1
    amp = amplification(m, toks, pv, tau, out_position(t), probes=1, iters=1)
    assert amp >= 0.0 and amp == amp, "JVP failed through the gated path"
