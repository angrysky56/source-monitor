"""Far-OOD: factual QA over a curated in-repo fact bank (no network).

Tests the signal on real-world knowledge (where the sibling L1 token probe
collapsed to ~.50). A trace states a couple of unrelated facts, then asks a
target question; the claim is the answer. Value-correct traces use real facts
(distractors = same-category real items); negation-correct traces use genuinely
unanswerable questions where "no reliable record" is the correct claim.
"""

from __future__ import annotations

import random

from source_monitor.llm.ood.base import OODClaim, OODTrace
from source_monitor.llm.task_render import Turn

SYSTEM = (
    "You answer general-knowledge questions concisely. If a question cannot be "
    "answered reliably, say you have no reliable record."
)
NEG_CONTENT = "I have no reliable record of that."
NEG_VALUE = "no reliable record"

# category -> list of (question, answer, [same-category distractors])
FACTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "capital": [
        ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Nice"]),
        ("What is the capital of Japan?", "Tokyo", ["Osaka", "Kyoto", "Nagoya"]),
        ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza", "Luxor"]),
        ("What is the capital of Canada?", "Ottawa", ["Toronto", "Montreal", "Vancouver"]),
        ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ],
    "element": [
        ("What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go"]),
        ("What is the chemical symbol for sodium?", "Na", ["So", "Sd", "Nm"]),
        ("What is the chemical symbol for iron?", "Fe", ["Ir", "In", "Fr"]),
        ("What is the chemical symbol for potassium?", "K", ["P", "Po", "Pt"]),
    ],
    "author": [
        ("Who wrote 'Hamlet'?", "Shakespeare", ["Marlowe", "Jonson", "Chaucer"]),
        ("Who wrote 'Pride and Prejudice'?", "Austen", ["Bronte", "Eliot", "Gaskell"]),
        ("Who wrote 'The Odyssey'?", "Homer", ["Virgil", "Ovid", "Hesiod"]),
    ],
    "planet": [
        ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Mars", "Earth"]),
        ("Which is the largest planet?", "Jupiter", ["Saturn", "Neptune", "Uranus"]),
    ],
}

UNANSWERABLE = [
    "What did Julius Caesar eat for breakfast on his tenth birthday?",
    "What is the favorite color of the 400th person born in 1850?",
    "How many grains of sand were on Brighton beach in 1783?",
    "What was the exact thought of Newton at noon on 3 March 1687?",
]


def _flat_facts() -> list[tuple[str, str, str, list[str]]]:
    return [(cat, q, a, d) for cat, items in FACTS.items() for (q, a, d) in items]


def generate(seed: int, n: int, n_context: int = 2) -> list[OODTrace]:
    rng = random.Random(seed)
    flat = _flat_facts()
    traces: list[OODTrace] = []
    for _ in range(n):
        turns = [Turn(role="system", content=SYSTEM, is_self=False, step_index=None)]
        # a couple of unrelated context Q/A (genuine)
        ctx = rng.sample(flat, k=min(n_context, len(flat)))
        step = 0
        for (_, q, a, _d) in ctx:
            turns.append(Turn(role="user", content=q, is_self=False, step_index=step))
            turns.append(Turn(role="assistant", content=f"{a}.", is_self=True, step_index=step))
            step += 1

        negation_correct = rng.random() < 0.3
        if negation_correct:
            q = rng.choice(UNANSWERABLE)
            contents = [NEG_CONTENT]
            values = [NEG_VALUE]
            surfaces = ["negation"]
            # add a couple of plausible-but-wrong concrete guesses as value distractors
            for guess in rng.sample(["Tuesday", "blue", "about 4000", "42"], k=2):
                contents.insert(0, f"It was {guess}.")
                values.insert(0, guess)
                surfaces.insert(0, "value")
            correct = len(contents) - 1  # negation is last
        else:
            cat, q, a, distractors = rng.choice(flat)
            picks = rng.sample(distractors, k=min(3, len(distractors)))
            opts = [a] + picks
            rng.shuffle(opts)
            contents = [f"{o}." for o in opts] + [NEG_CONTENT]
            values = list(opts) + [NEG_VALUE]
            surfaces = ["value"] * len(opts) + ["negation"]
            correct = opts.index(a)

        turns.append(Turn(role="user", content=q, is_self=False, step_index=step))
        turns.append(Turn(role="assistant", content=contents[correct], is_self=True,
                          step_index=step, is_corrupted=False,
                          claim_surface=surfaces[correct], location_text=values[correct]))
        claim = OODClaim(
            turn_index=len(turns) - 1,
            correct_index=correct,
            emitted_index=correct,
            candidate_contents=contents,
            candidate_values=values,
            candidate_surfaces=surfaces,
        )
        traces.append(OODTrace(domain="factual_qa", turns=turns, claim=claim,
                               meta={"negation_correct": negation_correct}))
    return traces
