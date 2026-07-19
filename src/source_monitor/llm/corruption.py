"""Text-level corruption injection for LLM traces.

Ports the three corruption types from source_monitor.task (token-level) to
operate on rendered Trace objects (text-level spans). Each injector returns
a CorruptionRecord or None if the trace has no eligible step.

A2: Records original and corrupted surface types for matched-surface stratification.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source_monitor.task import Task
    from source_monitor.llm.task_render import Trace, Turn

from source_monitor.task import REMOVE, NOWHERE, N_CONTAINERS, CON0
from source_monitor.llm.task_render import CONTAINER_NAMES, con_name


@dataclass
class CorruptionRecord:
    """Metadata for one corruption applied to a trace."""
    corruption_type: str            # "ghost" | "mislocation" | "phantom"
    step_index: int                 # which step was corrupted
    original_content: str           # the genuine assistant response
    corrupted_content: str          # the planted false response
    original_surface: str           # "container" | "nowhere" — A2
    corrupted_surface: str          # "container" | "nowhere" — A2
    trace: Trace                    # the full trace with corruption applied


def _corrupt_trace_turn(
    trace: Trace,
    step_index: int,
    new_loc_tok: int,
    corruption_type: str,
) -> CorruptionRecord:
    """Helper to return a CorruptionRecord with a corrupted copy of the trace."""
    # Deep copy the trace and the underlying task
    corrupt_trace = deepcopy(trace)
    task = corrupt_trace.task
    
    # 1. Update the underlying Task tokens and loc_targets
    emit_pos = task.emit_marker_pos[step_index]
    task.tokens[emit_pos + 1] = new_loc_tok
    task.loc_targets[step_index] = new_loc_tok
    
    # 2. Identify the assistant turn for this step in the Trace
    # turns layout: system (0), user (1), assistant (2), user (3), assistant (4)...
    # assistant turn for step_index k is at index 1 + k * 2 + 1 = 2 * (k + 1)
    turn_idx = 2 * (step_index + 1)
    turn = corrupt_trace.turns[turn_idx]
    assert turn.is_self and turn.role == "assistant"
    
    original_content = turn.content
    original_surface = turn.claim_surface
    assert original_surface is not None
    
    # 3. Construct new content and update turn fields
    q_name = corrupt_trace.query_object
    if new_loc_tok == NOWHERE:
        new_loc_str = "nowhere"
        new_content = f"The {q_name} is nowhere."
        corrupted_surface = "nowhere"
    else:
        new_loc_str = con_name(new_loc_tok)
        new_content = f"The {q_name} is in {new_loc_str}."
        corrupted_surface = "container"
        
    turn.content = new_content
    turn.is_corrupted = True
    turn.claim_surface = corrupted_surface
    turn.location_text = new_loc_str
    
    return CorruptionRecord(
        corruption_type=corruption_type,
        step_index=step_index,
        original_content=original_content,
        corrupted_content=new_content,
        original_surface=original_surface,
        corrupted_surface=corrupted_surface,
        trace=corrupt_trace,
    )


def _trace_containers(task) -> list[int]:
    """Container tokens actually named in this trace's task.

    Corruptions must draw the false container from this in-universe set. Drawing
    from the full global pool injects boxes that never appear in-context, adding
    a surface-novelty confound (an unseen box is surprising regardless of state)
    on top of the state-contradiction signal the detector is meant to measure.
    """
    return sorted({t for t in task.tokens if CON0 <= t < CON0 + N_CONTAINERS})


def inject_ghost_text(trace: Trace, rng: random.Random) -> CorruptionRecord | None:
    """Trained-on corruption type (the sps-blindspot ghost):

    Pick a step where the query object was just REMOVEd and rewrite the
    emitted loc from NOWHERE to a random container.
    """
    task = trace.task
    ghost_steps = [
        k for k, kind in enumerate(task.op_kinds)
        if kind == REMOVE and task.loc_targets[k] == NOWHERE
    ]
    if not ghost_steps:
        return None
    k = rng.choice(ghost_steps)
    used = _trace_containers(task)
    new_loc_tok = rng.choice(used)
    return _corrupt_trace_turn(trace, k, new_loc_tok, "ghost")


def inject_mislocation_text(trace: Trace, rng: random.Random) -> CorruptionRecord | None:
    """HELD-OUT corruption type (eval only):

    Pick a non-final step where the query object IS present and rewrite its
    emitted container to a different container.
    """
    task = trace.task
    # Non-final step where object is present
    steps = [
        k for k in range(len(task.loc_targets) - 1)
        if task.loc_targets[k] != NOWHERE
    ]
    if not steps:
        return None
    k = rng.choice(steps)
    cur = task.loc_targets[k]
    options = [c for c in _trace_containers(task) if c != cur]
    if not options:
        return None
    new_loc_tok = rng.choice(options)
    return _corrupt_trace_turn(trace, k, new_loc_tok, "mislocation")


def inject_phantom_text(trace: Trace, rng: random.Random) -> CorruptionRecord | None:
    """THIRD corruption type (eval only):

    Pick a non-final step where the query object IS present and rewrite its
    emitted location to NOWHERE.
    """
    task = trace.task
    steps = [
        k for k in range(len(task.loc_targets) - 1)
        if task.loc_targets[k] != NOWHERE
    ]
    if not steps:
        return None
    k = rng.choice(steps)
    return _corrupt_trace_turn(trace, k, NOWHERE, "phantom")


def inject_all_types(
    trace: Trace, rng: random.Random,
) -> dict[str, CorruptionRecord | None]:
    """Apply each corruption type to independent copies of the trace.

    Returns dict mapping corruption_type -> CorruptionRecord or None.
    """
    # Use independent sub-RNGs to avoid order of injection affecting choices
    rng_ghost = random.Random(rng.randint(0, 1000000))
    rng_misloc = random.Random(rng.randint(0, 1000000))
    rng_phantom = random.Random(rng.randint(0, 1000000))
    
    return {
        "ghost": inject_ghost_text(trace, rng_ghost),
        "mislocation": inject_mislocation_text(trace, rng_misloc),
        "phantom": inject_phantom_text(trace, rng_phantom),
    }
