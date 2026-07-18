"""
Entity-tracking task with dense per-step state emission + provenance labels.

Vendored from `sps-blindspot` (itself in the spirit of
`entity-tracking-externalization`): PUT/MOVE/REMOVE over objects and containers,
with the tracked object's location emitted after every op. New here:

  * provenance ids — which tokens are SELF-emitted (the fed-back `loc` tokens)
    vs EXTERNAL (ops, markers, query). A generating system always knows which
    tokens it produced itself; this file makes that structural fact explicit so
    the model can be given it.
  * inject_mislocation — a second, held-out corruption type (object present,
    wrong container) used ONLY at eval time to test whether a trained gate
    detects corruption *kinds* it never saw in training (the transfer question).

Sequence layout (fixed length for fixed n_ops):
    BOS  QRY qobj   [ OP OBJ CON  EMIT loc ]*n   EOS
The model predicts `loc` at each EMIT marker; the loc token that follows is the
model's own (teacher-forced) emission — the SELF-provenance channel.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ---- vocabulary (small, fixed) --------------------------------------------
PAD, BOS, EOS, QRY, ANS, NOWHERE, NONE, PUT, MOVE, REMOVE, EMIT = range(11)
N_OBJECTS = 15
N_CONTAINERS = 8
OBJ0 = 11
CON0 = OBJ0 + N_OBJECTS
VOCAB_SIZE = CON0 + N_CONTAINERS

STEP = 5  # tokens per op block: OP OBJ CON EMIT loc
HEAD = 3  # BOS QRY qobj


def obj_tok(i: int) -> int:
    return OBJ0 + i


def con_tok(i: int) -> int:
    return CON0 + i


@dataclass
class Task:
    tokens: list[int]
    emit_marker_pos: list[int]   # indices of EMIT markers (predict loc from here)
    loc_targets: list[int]       # loc token as WRITTEN in `tokens` (corrupted if ghosted)
    answer: int                  # true final loc token (never corrupted)
    query_obj: int
    n_removes: int
    query_obj_removed: bool
    final_absent: bool           # answer == NOWHERE
    op_kinds: list[int]          # op token per step (for choosing ghost positions)


def self_positions(task: Task) -> list[int]:
    """Indices of SELF-emitted tokens: the loc token after each EMIT marker."""
    return [p + 1 for p in task.emit_marker_pos]


def provenance_ids(task: Task, pad_to: int | None = None) -> list[int]:
    """Per-token origin: 0 = external (ops/markers/query/pad), 1 = self-emitted."""
    n = pad_to or len(task.tokens)
    ids = [0] * n
    for p in self_positions(task):
        ids[p] = 1
    return ids


def generate_task(
    rng: random.Random,
    n_ops: int = 8,
    n_objects: int = 4,
    n_containers: int = 3,
    remove_prob: float = 0.25,
) -> Task:
    """One random trace. Same generative process as sps-blindspot (vendored)."""
    objects = rng.sample(range(N_OBJECTS), k=min(n_objects, N_OBJECTS))
    containers = rng.sample(range(N_CONTAINERS), k=min(n_containers, N_CONTAINERS))
    location: dict[int, int | None] = {o: None for o in objects}
    q = rng.choice(objects)

    toks: list[int] = [BOS, QRY, obj_tok(q)]
    emit_marker_pos: list[int] = []
    loc_targets: list[int] = []
    op_kinds: list[int] = []
    n_removes = 0
    q_removed = False

    for _ in range(n_ops):
        present = [o for o in objects if location[o] is not None]
        absent = [o for o in objects if location[o] is None]
        roll = rng.random()
        if present and roll < remove_prob:
            o = rng.choice(present)
            toks += [REMOVE, obj_tok(o), NONE]
            location[o] = None
            n_removes += 1
            op_kinds.append(REMOVE)
            if o == q:
                q_removed = True
        elif present and roll < remove_prob + 0.35:
            o = rng.choice(present)
            dest = rng.choice([c for c in containers if c != location[o]] or containers)
            toks += [MOVE, obj_tok(o), con_tok(dest)]
            location[o] = dest
            op_kinds.append(MOVE)
        else:
            o = rng.choice(absent or objects)
            dest = rng.choice(containers)
            toks += [PUT, obj_tok(o), con_tok(dest)]
            location[o] = dest
            op_kinds.append(PUT)

        loc = NOWHERE if location[q] is None else con_tok(location[q])
        emit_marker_pos.append(len(toks))      # EMIT marker index
        toks += [EMIT, loc]
        loc_targets.append(loc)

    toks.append(EOS)
    return Task(
        tokens=toks,
        emit_marker_pos=emit_marker_pos,
        loc_targets=loc_targets,
        answer=loc_targets[-1],
        query_obj=obj_tok(q),
        n_removes=n_removes,
        query_obj_removed=q_removed,
        final_absent=loc_targets[-1] == NOWHERE,
        op_kinds=op_kinds,
    )


def generate_dataset(
    n_tasks: int,
    base_seed: int = 7,
    balance_absent: bool = True,
    **kw,
) -> list[Task]:
    """Reproducible task list; ~half end with the query object absent when balanced."""
    rng = random.Random(base_seed)
    out: list[Task] = []
    want_absent = True
    attempts = 0
    while len(out) < n_tasks and attempts < n_tasks * 50:
        attempts += 1
        t = generate_task(rng, **kw)
        if balance_absent and t.final_absent != want_absent:
            continue
        out.append(t)
        want_absent = not want_absent
    return out


def pad_batch(tasks: list[Task], pad_to: int | None = None) -> tuple[list[list[int]], list[int]]:
    lengths = [len(t.tokens) for t in tasks]
    L = pad_to or max(lengths)
    return [t.tokens + [PAD] * (L - len(t.tokens)) for t in tasks], lengths


def _with_corrupted_emission(task: Task, k: int, new_loc: int) -> Task:
    """Copy of `task` whose k-th emitted loc token is rewritten to `new_loc`."""
    toks = list(task.tokens)
    toks[task.emit_marker_pos[k] + 1] = new_loc
    corrupted = list(task.loc_targets)
    corrupted[k] = new_loc
    return Task(
        tokens=toks,
        emit_marker_pos=task.emit_marker_pos,
        loc_targets=corrupted,
        answer=task.answer,               # TRUE answer unchanged
        query_obj=task.query_obj,
        n_removes=task.n_removes,
        query_obj_removed=task.query_obj_removed,
        final_absent=task.final_absent,
        op_kinds=task.op_kinds,
    )


def inject_ghost(task: Task, rng: random.Random) -> Task | None:
    """
    Trained-on corruption type (the sps-blindspot ghost): pick a step where the
    query object was just REMOVEd and rewrite the emitted loc from NOWHERE to a
    random container — the model 'says' the removed object is still present.
    The true final answer is unchanged; the test is recovery.
    Returns None if no removal of the query object exists to ghost.
    """
    ghost_steps = [k for k, kind in enumerate(task.op_kinds)
                   if kind == REMOVE and task.loc_targets[k] == NOWHERE]
    if not ghost_steps:
        return None
    k = rng.choice(ghost_steps)
    return _with_corrupted_emission(task, k, con_tok(rng.randrange(N_CONTAINERS)))


def inject_mislocation(task: Task, rng: random.Random) -> Task | None:
    """
    HELD-OUT corruption type (eval only, never trained on): pick a non-final
    step where the query object IS present and rewrite its emitted container to
    a different container — a plausible-looking but false self-report. Tests
    whether a gate trained only on REMOVE-ghosts detects a corruption *kind* it
    never saw (contradiction-reading vs pattern-memorizing).
    Returns None if no eligible step exists.
    """
    steps = [k for k in range(len(task.loc_targets) - 1)
             if task.loc_targets[k] != NOWHERE]
    if not steps:
        return None
    k = rng.choice(steps)
    cur = task.loc_targets[k]
    options = [con_tok(c) for c in range(N_CONTAINERS) if con_tok(c) != cur]
    return _with_corrupted_emission(task, k, rng.choice(options))


def inject_phantom_removal(task: Task, rng: random.Random) -> Task | None:
    """
    THIRD corruption type (eval only): pick a non-final step where the query
    object IS present and rewrite its emitted location to NOWHERE — a phantom
    removal (the model 'says' a present object is gone). Completes the
    transfer matrix: ghost (absent->container), mislocation
    (container->other container), phantom (container->NOWHERE).
    Returns None if no eligible step exists.
    """
    steps = [k for k in range(len(task.loc_targets) - 1)
             if task.loc_targets[k] != NOWHERE]
    if not steps:
        return None
    return _with_corrupted_emission(task, rng.choice(steps), NOWHERE)
