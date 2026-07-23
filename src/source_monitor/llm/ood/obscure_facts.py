"""ood/obscure_facts.py — Build B substrate: obscure factual QA (confabulation zone).

Seed bank for the regime F26/F27 never tested: facts Qwen3-1.7B HALF-knows, so it
answers confidently, is often wrong, and its wrong answers VARY across samples —
where sampling variance (Leg 2) is the only signal and surprisal is blind. See
`BUILD-B-CONFABULATION.md`.

⚠ VERIFICATION REQUIRED. These ground-truth answers are LLM-authored — a
hallucination risk in a hallucination-detection project, so spot-check against an
authoritative source (Wikipedia / PubChem) before trusting any F28 number.
Deliberate choices to reduce that risk:
- Only STABLE facts: element symbols, atomic numbers, long-standing capitals,
  currencies. No populations, GDP, office-holders, or dates.
- Contested / renamed capitals EXCLUDED on purpose (no Kazakhstan, Bolivia,
  Tanzania, Ivory Coast, Sri Lanka, Myanmar) — single unambiguous seat only.
- Chemical symbols/atomic numbers are the most reliable rows; capitals/currencies
  next. If any row is wrong, fix it — a mislabeled positive/negative poisons F28.

Design:
- Neutral SYSTEM prompt: it does NOT invite abstention (unlike factual_qa's, which
  actively offers "no reliable record" and pushed F26 into the refusal zone). This
  keeps the model ATTEMPTING answers → the confabulation zone.
- Ground truth for labeling lives in `meta["acceptable"]` (lowercased canonical +
  variants). The F28 runner MUST label correctness against that list with
  word-boundary matching (single/short answers like "W", "27" false-match as
  substrings — the F26 negation-bias trap in a new place). NOT via
  `sampled_consistency`'s single-value correct_rate.
- This is a STARTING POINT, not a validated confabulation set. The §4 calibration
  in the spec filters it into Known / Confabulation / Refusal zones per model, by
  eye, before any AUROC is trusted.
"""

from __future__ import annotations

import random

from source_monitor.llm.ood.base import OODClaim, OODTrace
from source_monitor.llm.task_render import Turn

SYSTEM = (
    "You are a knowledgeable assistant. Answer each question with a short, direct "
    "answer of just a few words."
)

# (category, question, canonical_answer, [accepted variants incl. canonical])
FACTS: list[tuple[str, str, str, list[str]]] = [
    # --- element symbols (mostly 2-letter to reduce match fragility) ---
    ("symbol", "What is the chemical symbol for antimony?", "Sb", ["sb"]),
    ("symbol", "What is the chemical symbol for tungsten?", "W", ["w", "wolfram"]),
    ("symbol", "What is the chemical symbol for tin?", "Sn", ["sn"]),
    ("symbol", "What is the chemical symbol for lead?", "Pb", ["pb"]),
    ("symbol", "What is the chemical symbol for bismuth?", "Bi", ["bi"]),
    ("symbol", "What is the chemical symbol for molybdenum?", "Mo", ["mo"]),
    ("symbol", "What is the chemical symbol for cadmium?", "Cd", ["cd"]),
    ("symbol", "What is the chemical symbol for zirconium?", "Zr", ["zr"]),
    ("symbol", "What is the chemical symbol for palladium?", "Pd", ["pd"]),
    ("symbol", "What is the chemical symbol for tantalum?", "Ta", ["ta"]),
    ("symbol", "What is the chemical symbol for niobium?", "Nb", ["nb"]),
    ("symbol", "What is the chemical symbol for osmium?", "Os", ["os"]),
    ("symbol", "What is the chemical symbol for rhenium?", "Re", ["re"]),
    ("symbol", "What is the chemical symbol for tellurium?", "Te", ["te"]),
    ("symbol", "What is the chemical symbol for mercury?", "Hg", ["hg"]),
    ("symbol", "What is the chemical symbol for silver?", "Ag", ["ag"]),
    # --- atomic numbers (distinctive numeric answers) ---
    ("atomic_number", "What is the atomic number of cobalt?", "27", ["27"]),
    ("atomic_number", "What is the atomic number of nickel?", "28", ["28"]),
    ("atomic_number", "What is the atomic number of zinc?", "30", ["30"]),
    ("atomic_number", "What is the atomic number of bromine?", "35", ["35"]),
    ("atomic_number", "What is the atomic number of silver?", "47", ["47"]),
    ("atomic_number", "What is the atomic number of iodine?", "53", ["53"]),
    ("atomic_number", "What is the atomic number of barium?", "56", ["56"]),
    ("atomic_number", "What is the atomic number of platinum?", "78", ["78"]),
    ("atomic_number", "What is the atomic number of mercury?", "80", ["80"]),
    ("atomic_number", "What is the atomic number of lead?", "82", ["82"]),
    # --- capitals (single unambiguous seat; contested ones excluded) ---
    ("capital", "What is the capital of Mongolia?", "Ulaanbaatar", ["ulaanbaatar", "ulan bator"]),
    ("capital", "What is the capital of Bhutan?", "Thimphu", ["thimphu"]),
    ("capital", "What is the capital of Laos?", "Vientiane", ["vientiane"]),
    ("capital", "What is the capital of Cambodia?", "Phnom Penh", ["phnom penh"]),
    ("capital", "What is the capital of Brunei?", "Bandar Seri Begawan", ["bandar seri begawan"]),
    ("capital", "What is the capital of Suriname?", "Paramaribo", ["paramaribo"]),
    ("capital", "What is the capital of Paraguay?", "Asuncion", ["asuncion", "asunción"]),
    ("capital", "What is the capital of Uruguay?", "Montevideo", ["montevideo"]),
    ("capital", "What is the capital of Kyrgyzstan?", "Bishkek", ["bishkek"]),
    ("capital", "What is the capital of Tajikistan?", "Dushanbe", ["dushanbe"]),
    ("capital", "What is the capital of Turkmenistan?", "Ashgabat", ["ashgabat"]),
    ("capital", "What is the capital of Uzbekistan?", "Tashkent", ["tashkent"]),
    ("capital", "What is the capital of Armenia?", "Yerevan", ["yerevan"]),
    ("capital", "What is the capital of Azerbaijan?", "Baku", ["baku"]),
    ("capital", "What is the capital of Moldova?", "Chisinau", ["chisinau", "chișinău"]),
    ("capital", "What is the capital of Slovenia?", "Ljubljana", ["ljubljana"]),
    ("capital", "What is the capital of Latvia?", "Riga", ["riga"]),
    ("capital", "What is the capital of Estonia?", "Tallinn", ["tallinn"]),
    ("capital", "What is the capital of Rwanda?", "Kigali", ["kigali"]),
    ("capital", "What is the capital of Senegal?", "Dakar", ["dakar"]),
    ("capital", "What is the capital of Mali?", "Bamako", ["bamako"]),
    ("capital", "What is the capital of Burkina Faso?", "Ouagadougou", ["ouagadougou"]),
    ("capital", "What is the capital of Madagascar?", "Antananarivo", ["antananarivo"]),
    ("capital", "What is the capital of Zambia?", "Lusaka", ["lusaka"]),
    ("capital", "What is the capital of Botswana?", "Gaborone", ["gaborone"]),
    ("capital", "What is the capital of Namibia?", "Windhoek", ["windhoek"]),
    ("capital", "What is the capital of Mozambique?", "Maputo", ["maputo"]),
    # --- currencies (stable) ---
    ("currency", "What is the currency of Thailand?", "baht", ["baht"]),
    ("currency", "What is the currency of Vietnam?", "dong", ["dong"]),
    ("currency", "What is the currency of Poland?", "zloty", ["zloty", "złoty"]),
    ("currency", "What is the currency of Hungary?", "forint", ["forint"]),
    ("currency", "What is the currency of Malaysia?", "ringgit", ["ringgit"]),
    ("currency", "What is the currency of Nigeria?", "naira", ["naira"]),
    ("currency", "What is the currency of Peru?", "sol", ["sol", "nuevo sol"]),
]


def generate(seed: int, n: int, **kwargs) -> list[OODTrace]:
    """Yield up to n obscure-fact traces (system + question), ground truth in meta.

    Each trace has no assistant turn — `claim.turn_index` points past the question
    so `consistency.sample_answers` prompts the model to answer. Correctness must be
    labeled by the runner against `meta["acceptable"]`, not the claim's candidates.
    """
    rng = random.Random(seed)
    pool = FACTS[:]
    rng.shuffle(pool)
    items = pool[: n] if n < len(pool) else pool

    traces: list[OODTrace] = []
    for cat, q, ans, accepted in items:
        turns = [
            Turn(role="system", content=SYSTEM, is_self=False, step_index=None),
            Turn(role="user", content=q, is_self=False, step_index=0),
        ]
        claim = OODClaim(
            turn_index=len(turns),  # no assistant turn: prompt ends at the question
            correct_index=0,
            emitted_index=0,
            candidate_contents=[ans],
            candidate_values=[ans],
            candidate_surfaces=["value"],
        )
        traces.append(
            OODTrace(
                domain="obscure_facts",
                turns=turns,
                claim=claim,
                meta={
                    "category": cat,
                    "acceptable": [a.lower() for a in accepted],
                    "canonical": ans,
                },
            )
        )
    return traces
