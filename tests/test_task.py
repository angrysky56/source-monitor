"""Task-structure, provenance, and corruption-injection invariants."""

from __future__ import annotations

import random

from source_monitor.task import (
    EMIT,
    HEAD,
    NOWHERE,
    STEP,
    generate_dataset,
    inject_ghost,
    inject_mislocation,
    provenance_ids,
    self_positions,
)

TKW = dict(n_ops=8, n_objects=4, n_containers=3)


def test_fixed_layout_and_provenance():
    tasks = generate_dataset(64, base_seed=7, **TKW)
    L = len(tasks[0].tokens)
    for t in tasks:
        assert len(t.tokens) == L, "fixed n_ops must give fixed length"
        # EMIT marker of step k sits at HEAD + k*STEP + 3
        for k, p in enumerate(t.emit_marker_pos):
            assert p == HEAD + k * STEP + 3
            assert t.tokens[p] == EMIT
        ids = provenance_ids(t)
        sp = set(self_positions(t))
        for i, v in enumerate(ids):
            assert v == (1 if i in sp else 0)
        # self tokens are exactly the emitted locs
        for k, p in enumerate(self_positions(t)):
            assert t.tokens[p] == t.loc_targets[k]


def test_dataset_deterministic():
    a = generate_dataset(32, base_seed=11, **TKW)
    b = generate_dataset(32, base_seed=11, **TKW)
    assert all(x.tokens == y.tokens for x, y in zip(a, b))


def _one_diff(a: list[int], b: list[int]) -> int:
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diffs) == 1, f"expected exactly one differing token, got {diffs}"
    return diffs[0]


def test_inject_ghost_invariants():
    rng = random.Random(0)
    tasks = [t for t in generate_dataset(200, base_seed=3, balance_absent=False, **TKW)
             if t.query_obj_removed]
    assert tasks, "need ghost-able tasks"
    n_done = 0
    for t in tasks:
        g = inject_ghost(t, rng)
        if g is None:
            continue
        n_done += 1
        i = _one_diff(t.tokens, g.tokens)
        k = _one_diff(t.loc_targets, g.loc_targets)
        assert i == t.emit_marker_pos[k] + 1          # only the emitted loc changed
        assert t.loc_targets[k] == NOWHERE            # ghost replaces a true NOWHERE
        assert g.loc_targets[k] != NOWHERE
        assert g.answer == t.answer                   # true answer preserved
    assert n_done > 0


def test_inject_mislocation_invariants():
    rng = random.Random(0)
    n_done = 0
    for t in generate_dataset(200, base_seed=5, balance_absent=False, **TKW):
        m = inject_mislocation(t, rng)
        if m is None:
            continue
        n_done += 1
        i = _one_diff(t.tokens, m.tokens)
        k = _one_diff(t.loc_targets, m.loc_targets)
        assert i == t.emit_marker_pos[k] + 1
        assert t.loc_targets[k] != NOWHERE            # object was present
        assert m.loc_targets[k] != t.loc_targets[k]   # different container
        assert k < len(t.loc_targets) - 1             # never the final emission
        assert m.answer == t.answer
    assert n_done > 0
