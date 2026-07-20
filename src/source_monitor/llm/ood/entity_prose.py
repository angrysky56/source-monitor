"""Near-OOD bridge: entity tracking rendered as free prose (no rigid template).

Same underlying task as Phase 0 (place/move/remove objects, query final
location) but the CONTEXT turns use a varied phrasing pool, so this isolates the
effect of dropping the fixed template before true OOD. Candidates for the final
claim are the trace's own in-context locations plus a negation ("isn't anywhere")
— the phantom-analog. No global pool → no Phase-0 surface-novelty confound.
"""

from __future__ import annotations

import random

from source_monitor.llm.ood.base import OODClaim, OODTrace
from source_monitor.llm.task_render import Turn

SYSTEM = (
    "You are tracking where objects are. After each update, if asked, report "
    "the object's current location, or say it isn't anywhere if it was removed."
)

OBJECTS = [
    "red ball", "blue mug", "green book", "steel key", "glass vial",
    "wooden spoon", "brass coin", "paper crane", "silk scarf", "clay pot",
]
LOCATIONS = [
    "the kitchen", "the attic", "the garage", "the study", "the cellar",
    "the pantry", "the hallway", "the balcony",
]

PUT_PHRASES = [
    "Put the {o} in {l}.",
    "Place the {o} into {l}.",
    "Set the {o} down in {l}.",
    "Leave the {o} in {l}.",
]
MOVE_PHRASES = [
    "Now move the {o} to {l}.",
    "Take the {o} over to {l}.",
    "Relocate the {o} to {l}.",
]
ACK_PHRASES = [
    "Done — the {o} is in {l}.",
    "Okay, the {o} is now in {l}.",
    "Got it; the {o} sits in {l}.",
]
CLAIM_TEMPLATES = [
    "The {o} is in {l}.",
    "Right now the {o} is in {l}.",
    "Currently the {o} is in {l}.",
]
NEG_TEMPLATES = [
    "The {o} isn't anywhere anymore.",
    "The {o} is no longer anywhere.",
]
NEG_VALUES = ["isn't anywhere anymore", "no longer anywhere"]


def generate(seed: int, n: int, n_ops: int = 5, corrupt_mid: bool = False) -> list[OODTrace]:
    rng = random.Random(seed)
    traces: list[OODTrace] = []
    for _ in range(n):
        obj = rng.choice(OBJECTS)
        locs = rng.sample(LOCATIONS, k=rng.randint(3, 4))
        turns = [Turn(role="system", content=SYSTEM, is_self=False, step_index=None)]
        cur = rng.choice(locs)
        # initial placement
        turns.append(Turn(role="user", content=rng.choice(PUT_PHRASES).format(o=obj, l=cur),
                          is_self=False, step_index=0))
        turns.append(Turn(role="assistant", content=rng.choice(ACK_PHRASES).format(o=obj, l=cur),
                          is_self=True, step_index=0, claim_surface="container", location_text=cur))
        mentioned = [cur]  # locations that actually appear in-context (anti-confound)
        removed = False
        for k in range(1, n_ops):
            if rng.random() < 0.25 and not removed:
                turns.append(Turn(role="user", content=f"Take the {obj} away.",
                                  is_self=False, step_index=k))
                turns.append(Turn(role="assistant", content=f"Okay, the {obj} has been removed.",
                                  is_self=True, step_index=k))
                removed = True
            else:
                nxt = rng.choice([l for l in locs if l != cur] or locs)
                turns.append(Turn(role="user", content=rng.choice(MOVE_PHRASES).format(o=obj, l=nxt),
                                  is_self=False, step_index=k))
                turns.append(Turn(role="assistant", content=rng.choice(ACK_PHRASES).format(o=obj, l=nxt),
                                  is_self=True, step_index=k, claim_surface="container", location_text=nxt))
                cur = nxt
                if nxt not in mentioned:
                    mentioned.append(nxt)
                removed = False

        # Corruption-exposure (Phase 2): plant a false location in a non-final ack.
        # The true final location is unchanged, so the final claim stays correct;
        # this is a lie in-context that later emissions must re-derive past.
        corrupt_turn_index = None
        if corrupt_mid and len(mentioned) >= 2:
            ack_idxs = [i for i, t in enumerate(turns)
                        if t.role == "assistant" and t.location_text in mentioned]
            if ack_idxs:
                ti = rng.choice(ack_idxs)
                old = turns[ti].location_text
                wrong = rng.choice([m for m in mentioned if m != old])
                turns[ti].content = turns[ti].content.replace(old, wrong)
                turns[ti].is_corrupted = True
                turns[ti].location_text = wrong
                corrupt_turn_index = ti

        # Final query + claim (candidates = in-context locations + negation).
        claim_tmpl = rng.choice(CLAIM_TEMPLATES)
        neg_i = rng.randrange(len(NEG_TEMPLATES))
        contents = [claim_tmpl.format(o=obj, l=l) for l in mentioned] + [NEG_TEMPLATES[neg_i].format(o=obj)]
        values = list(mentioned) + [NEG_VALUES[neg_i]]
        surfaces = ["value"] * len(mentioned) + ["negation"]
        correct = len(mentioned) if removed else mentioned.index(cur)

        turns.append(Turn(role="user", content=f"Where is the {obj} now?",
                          is_self=False, step_index=n_ops))
        turns.append(Turn(role="assistant", content=contents[correct], is_self=True,
                          step_index=n_ops, is_corrupted=False,
                          claim_surface=surfaces[correct], location_text=values[correct]))

        claim = OODClaim(
            turn_index=len(turns) - 1,
            correct_index=correct,
            emitted_index=correct,
            candidate_contents=contents,
            candidate_values=values,
            candidate_surfaces=surfaces,
        )
        traces.append(OODTrace(domain="entity_prose", turns=turns, claim=claim,
                               meta={"removed": removed, "corrupt_turn_index": corrupt_turn_index}))
    return traces
