"""Convert the toy's entity-tracking task to multi-turn natural language chat traces.

A1: Ensure no template formatting is mixed into the scored text.
A4: Expose location_text and track which tokens are location-slot tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source_monitor.task import Task

from source_monitor.task import (
    PUT, MOVE, REMOVE, NOWHERE, OBJ0, CON0, N_CONTAINERS, N_OBJECTS
)

OBJECT_NAMES: tuple[str, ...] = (
    "red ball", "blue cube", "green cylinder", "yellow pyramid",
    "white sphere", "black cone", "orange disc", "purple ring",
    "silver rod", "bronze star", "pink shell", "grey stone",
    "teal prism", "coral wedge", "amber bead",
)

CONTAINER_NAMES: tuple[str, ...] = (
    "box A", "box B", "box C", "box D",
    "box E", "box F", "box G", "box H",
)

SYSTEM_PROMPT = (
    "You are tracking objects placed in containers. "
    "After each operation, report the current location of the tracked object. "
    "If it has been removed, say 'nowhere'."
)


@dataclass
class Turn:
    role: str               # "system" | "user" | "assistant"
    content: str
    is_self: bool           # True for assistant turns
    step_index: int | None  # 0-indexed step, None for system
    is_corrupted: bool = False
    claim_surface: str | None = None  # "container" | "nowhere" | None
    location_text: str | None = None  # e.g., "box A" or "nowhere"


@dataclass
class Trace:
    turns: list[Turn]
    query_object: str       # e.g., "red ball"
    ground_truth_final: str # e.g., "box C" or "nowhere"
    op_kinds: list[int]
    task: Task


def obj_name(token: int) -> str:
    """Map OBJ0+i to OBJECT_NAMES[i]."""
    idx = token - OBJ0
    if 0 <= idx < len(OBJECT_NAMES):
        return OBJECT_NAMES[idx]
    raise ValueError(f"Invalid object token: {token}")


def con_name(token: int) -> str:
    """Map CON0+i to CONTAINER_NAMES[i]."""
    idx = token - CON0
    if 0 <= idx < len(CONTAINER_NAMES):
        return CONTAINER_NAMES[idx]
    raise ValueError(f"Invalid container token: {token}")


def loc_name(token: int) -> str:
    """Map location token (NOWHERE or CON0+i) to text."""
    if token == NOWHERE:
        return "nowhere"
    return con_name(token)


def render_trace(task: Task) -> Trace:
    """Convert a token-level Task into a natural language Trace."""
    q_name = obj_name(task.query_obj)
    
    turns: list[Turn] = [
        Turn(
            role="system",
            content=SYSTEM_PROMPT,
            is_self=False,
            step_index=None,
        )
    ]
    
    # Parse tokens to reconstruct the user ops and assistant responses
    # Layout of Task: BOS QRY qobj [OP OBJ CON EMIT loc]*n EOS
    # We can reconstruct from task.emit_marker_pos, which points to the EMIT tokens.
    # The actual tokens before EMIT are: OP, OBJ, CON
    for i, emit_pos in enumerate(task.emit_marker_pos):
        op_tok = task.tokens[emit_pos - 3]
        obj_tok = task.tokens[emit_pos - 2]
        con_tok = task.tokens[emit_pos - 1]
        
        o_name = obj_name(obj_tok)
        
        # Build user turn
        if op_tok == PUT:
            c_name = con_name(con_tok)
            user_content = f"Put the {o_name} in {c_name}."
        elif op_tok == MOVE:
            c_name = con_name(con_tok)
            user_content = f"Move the {o_name} to {c_name}."
        elif op_tok == REMOVE:
            user_content = f"Remove the {o_name}."
        else:
            raise ValueError(f"Unknown operation token: {op_tok}")
            
        turns.append(
            Turn(
                role="user",
                content=user_content,
                is_self=False,
                step_index=i,
            )
        )
        
        # Build assistant turn
        loc_tok = task.tokens[emit_pos + 1]
        loc_str = loc_name(loc_tok)
        claim_surface = "nowhere" if loc_tok == NOWHERE else "container"
        
        if loc_tok == NOWHERE:
            assistant_content = f"The {q_name} is nowhere."
        else:
            assistant_content = f"The {q_name} is in {loc_str}."
            
        turns.append(
            Turn(
                role="assistant",
                content=assistant_content,
                is_self=True,
                step_index=i,
                claim_surface=claim_surface,
                location_text=loc_str,
            )
        )
        
    return Trace(
        turns=turns,
        query_object=q_name,
        ground_truth_final=loc_name(task.answer),
        op_kinds=list(task.op_kinds),
        task=task,
    )
