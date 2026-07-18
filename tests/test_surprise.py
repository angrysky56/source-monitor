"""v2 guardrails: surprisal-gate mechanics + phantom-removal injector."""

from __future__ import annotations

import random

import torch

from source_monitor.model import SMConfig, SMDecoder
from source_monitor.task import (
    NOWHERE,
    VOCAB_SIZE,
    generate_dataset,
    inject_phantom_removal,
    provenance_ids,
    self_positions,
)

TKW = dict(n_ops=4, n_objects=3, n_containers=2)


def tiny_cfg(**kw) -> SMConfig:
    base = dict(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2, n_layers=3,
                d_ff=64, gate_layer=2)
    base.update(kw)
    return SMConfig(**base)


def test_phantom_invariants():
    rng = random.Random(0)
    n_done = 0
    for t in generate_dataset(200, base_seed=9, balance_absent=False, **TKW):
        p = inject_phantom_removal(t, rng)
        if p is None:
            continue
        n_done += 1
        diffs = [i for i, (a, b) in enumerate(zip(t.tokens, p.tokens)) if a != b]
        assert len(diffs) == 1
        k = [i for i in range(len(t.loc_targets))
             if t.loc_targets[i] != p.loc_targets[i]][0]
        assert diffs[0] == t.emit_marker_pos[k] + 1
        assert t.loc_targets[k] != NOWHERE      # object was present
        assert p.loc_targets[k] == NOWHERE      # falsified to absent
        assert k < len(t.loc_targets) - 1
        assert p.answer == t.answer
    assert n_done > 0


def test_surprise_causality():
    """Two-pass mode must remain sequence-causal."""
    torch.manual_seed(0)
    tasks = generate_dataset(2, base_seed=1, **TKW)
    t = tasks[0]
    m = SMDecoder(tiny_cfg(gate="surprise", provenance=True))
    x = torch.tensor([t.tokens], dtype=torch.long)
    pv = torch.tensor([provenance_ids(t)], dtype=torch.long)
    j = t.emit_marker_pos[1]  # mid-sequence
    x2 = x.clone()
    x2[:, j + 1 :] = torch.tensor(tasks[1].tokens[j + 1 :], dtype=torch.long)
    with torch.no_grad():
        a, _ = m(x, pv)
        b, _ = m(x2, pv)
    assert torch.allclose(a[:, : j + 1], b[:, : j + 1], atol=1e-5), \
        "surprise gate leaked future information into past logits"


def test_surprise_gamma_only_on_self():
    torch.manual_seed(1)
    t = generate_dataset(1, base_seed=2, **TKW)[0]
    m = SMDecoder(tiny_cfg(gate="surprise", provenance=True))
    x = torch.tensor([t.tokens], dtype=torch.long)
    pv = torch.tensor([provenance_ids(t)], dtype=torch.long)
    with torch.no_grad():
        _, gamma = m(x, pv)
    assert gamma is not None
    sp = set(self_positions(t))
    for i in range(x.shape[1]):
        if i not in sp:
            assert float(gamma[0, i]) == 0.0, "gamma nonzero off self positions"


def test_surprise_external_noop():
    """All-external provenance: bit-exact equality with the ungated model path."""
    torch.manual_seed(2)
    t = generate_dataset(1, base_seed=3, **TKW)[0]
    m = SMDecoder(tiny_cfg(gate="surprise", provenance=False))
    x = torch.tensor([t.tokens], dtype=torch.long)
    ext = torch.zeros_like(x)
    with torch.no_grad():
        a, _ = m(x, ext)
        m.surp_b.fill_(-30.0)   # would slam every gate shut — but nothing is self
        b, _ = m(x, ext)
    assert torch.equal(a, b)


def test_surprise_gamma_wiring():
    """gamma = a*logp + b with logp <= 0 and a=1 at init, so gamma <= b at
    every self position. Verifies the wiring, not detection quality (that is
    the experiment)."""
    torch.manual_seed(3)
    t = generate_dataset(1, base_seed=4, **TKW)[0]
    m = SMDecoder(tiny_cfg(gate="surprise", provenance=True))
    x = torch.tensor([t.tokens], dtype=torch.long)
    pv = torch.tensor([provenance_ids(t)], dtype=torch.long)
    with torch.no_grad():
        _, gamma = m(x, pv)
    b = float(m.surp_b.detach())
    for p in self_positions(t):
        g = float(gamma[0, p])
        assert g == g and g <= b + 1e-4, "gamma wiring broken (expected a*logp+b <= b)"


def test_surprise_train_step():
    """One optimization step through the two-pass graph (incl. surp_a/b grads)."""
    torch.manual_seed(4)
    from source_monitor.train import batch_loss, build_bundle
    tasks = generate_dataset(24, base_seed=5, **TKW)
    bundle = build_bundle(tasks, torch.device("cpu"), seed=0)
    m = SMDecoder(tiny_cfg(gate="surprise", provenance=True))
    loss = batch_loss(m, bundle, torch.arange(8), ghost_frac=0.5, gate_loss_w=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.surp_a.grad is not None and torch.isfinite(m.surp_a.grad)
    assert m.surp_b.grad is not None and torch.isfinite(m.surp_b.grad)


def test_emission_dropout_mechanics():
    """A dropped self-emission must change later logits (it is masked) but not
    earlier ones (causality); and dropout in batch_loss must train finitely."""
    torch.manual_seed(5)
    t = generate_dataset(1, base_seed=6, **TKW)[0]
    m = SMDecoder(tiny_cfg(provenance=True))
    x = torch.tensor([t.tokens], dtype=torch.long)
    pv = torch.tensor([provenance_ids(t)], dtype=torch.long)
    p = self_positions(t)[1]
    drop = torch.zeros_like(x, dtype=torch.bool)
    drop[0, p] = True
    with torch.no_grad():
        a, _ = m(x, pv)
        b, _ = m(x, pv, drop=drop)
    assert torch.allclose(a[:, :p], b[:, :p], atol=1e-5)
    assert not torch.allclose(a[:, p:], b[:, p:], atol=1e-5), "drop had no effect"

    from source_monitor.train import batch_loss, build_bundle
    tasks = generate_dataset(24, base_seed=7, **TKW)
    bundle = build_bundle(tasks, torch.device("cpu"), seed=0)
    for gate in ("none", "surprise"):
        mm = SMDecoder(tiny_cfg(gate=gate, provenance=True))
        loss = batch_loss(mm, bundle, torch.arange(8), ghost_frac=0.0,
                          gate_loss_w=0.5, emission_dropout=0.5)
        assert torch.isfinite(loss)
        loss.backward()


def test_hard_gate_mechanics():
    """gate_hard: negative gamma must produce a much stronger eviction than
    soft mode; positive-side behavior unchanged; loss path finite."""
    torch.manual_seed(6)
    t = generate_dataset(1, base_seed=8, **TKW)[0]
    x = torch.tensor([t.tokens], dtype=torch.long)
    pv = torch.tensor([provenance_ids(t)], dtype=torch.long)

    soft_m = SMDecoder(tiny_cfg(gate="surprise", provenance=True))
    hard_m = SMDecoder(tiny_cfg(gate="surprise", provenance=True, gate_hard=True))
    hard_m.load_state_dict(soft_m.state_dict())
    with torch.no_grad():
        # open gates (gamma ~ b > 0): hard and soft must agree exactly
        a, ga = soft_m(x, pv)
        b, gb = hard_m(x, pv)
        if bool((ga[0, [p for p in self_positions(t)]] > 0).all()):
            assert torch.equal(a, b), "hard mode changed open-gate behavior"
        # force gamma negative everywhere: hard must diverge from soft
        soft_m.surp_b.fill_(-5.0)
        hard_m.surp_b.fill_(-5.0)
        c, _ = soft_m(x, pv)
        d, _ = hard_m(x, pv)
    assert not torch.allclose(c, d, atol=1e-5), "hard eviction had no extra effect"

    from source_monitor.train import batch_loss, build_bundle
    tasks = generate_dataset(24, base_seed=9, **TKW)
    bundle = build_bundle(tasks, torch.device("cpu"), seed=0)
    loss = batch_loss(hard_m, bundle, torch.arange(8), ghost_frac=0.3, gate_loss_w=0.5)
    assert torch.isfinite(loss)
    loss.backward()
