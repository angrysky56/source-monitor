# Build B — the confabulation task (the regime that justifies Leg 2)

**⚠ REORIENTED by the F28-prep probes (2026-07-23). Read this first.** The two
calibration probes this spec called for were run, and they narrowed Leg 2's role
sharply: **self-consistency detects instability, not wrongness.** The model's
confident systematic errors are STABLE (e.g. "35×85" → "3025" six times), so
sampling variance is blind to them — that is Class 3, and it needs Leg 3 (external
grounding), not Leg 2. Consequences for this spec:
- Leg 2's real niche is NARROW: uncertainty-driven flailing (obscure recall the
  model is unsure of), not systematic error. A Build B eval that demonstrates this
  is still worth having, but it is a *measurement of a narrow niche*, not the
  headline — the headline moved to `BUILD-C-GROUNDING.md`.
- Before any Build B eval: **fix `distinct_ratio`** — it conflates surface/phrasing
  variation with real uncertainty (probe 1: a correct-but-verbose answer scored
  dr .67 and would false-flag). Cluster answers semantically or anchor on
  correctness, not string-distinctness.
- Positives must be UNSTABLE confabulations (the only kind Leg 2 can catch), sourced
  from external-verified obscure recall — not math (math errors are stable) and not
  the current too-easy bank (`ood/obscure_facts.py` is now a Known-*negatives* set).

**Original status:** co-designed spec (Ty + reviewer), 2026-07-23. **Not a
cold-agent hand-off** — §4 (calibration) requires a human reading raw samples.
Companion: `BUILD-HANDOFF.md` §3, `FINDINGS.md` F26/F27/F28-prep,
`BUILD-C-GROUNDING.md`.

---

## 1. Why this task must exist

F27 validated the router *mechanism* but couldn't validate its *value on quality*,
because both existing tasks put Leg 2 in the wrong regime:
- **entity_prose / arithmetic (F25):** the lie contradicts context → surprisal
  already near-perfect → Leg 2 unneeded.
- **factual_qa planted lies (F27):** wrong values on *known* facts → a wrong
  known-fact value is surprising even without context → surprisal catches it too.
- **factual_qa unanswerable (F26):** *obviously* unanswerable → the model
  correctly abstains → the signal is calibrated refusal (hedge-rate), not variance.

Leg 2's *unique* home — the only place `distinct_ratio` (sampling variance) is the
sole signal — has never been benchmarked. That regime is **confident confabulation
on facts the model half-knows**: it does not abstain (thinks it knows), it is
often wrong, and its wrong answers *vary across independent samples*. Surprisal is
blind here by construction (a confabulation is low-surprisal *to the model that
produced it* — it believes its own guess). This is exactly the real-world
hallucination seer exists to catch, and the thing F26d only half-tested.

---

## 2. The three-zone problem (the knife-edge to design against)

Every candidate question lands the model in one of three zones. Only the middle
one is useful, and it is a narrow target:

| Zone | Model behavior | Signal present | Verdict |
|------|----------------|----------------|---------|
| **Known** | confident, correct, stable across samples | none needed | negative (don't flag) |
| **Confabulation** ← TARGET | confident, wrong, **varies** across samples | `distinct_ratio` (variance); NOT surprisal, NOT hedge | positive (flag) |
| **Refusal** | abstains / "no reliable record", stably | hedge-rate (F26), not variance | out of scope — drifts back to F26 |

The whole design problem is **staying in the middle column**: questions obscure
enough that the model is wrong and unstable, but *plausible enough that it
attempts an answer instead of refusing.* Too easy → Known (F27's failure). Too
obviously-impossible → Refusal (F26's failure).

---

## 3. Substrates

Build both; they stress the claim from opposite sides.

### 3a. Primary — curated obscure facts (`ood/obscure_facts.py`)
Hand-curated real questions with **external ground truth**, chosen to be obscure
enough that Qwen3-1.7B half-knows them: e.g. capitals of small nations, atomic
numbers of less-common elements, birth years of minor historical figures,
directors of non-famous films, populations, discovery dates. ~40–80 items, each
`(question, ground_truth, category)`.
- **Why primary:** this is where surprisal is *known-weak* (F19: factual recall
  surprisal ~chance), so it's the fair test of Leg 2's unique value.
- **Why it needs calibration (§4):** which items land in the confabulation zone is
  a property of *this model* and can't be known a priori.

### 3b. Clean control — hard arithmetic (extend `ood/arithmetic.py` or new)
Auto-generated products the model cannot compute in-head, thinking off, e.g.
`17 × 23 × 19`, with **Python ground truth**. Guarantees the model is uncertain
(no memorized answer) and produces *varying* wrong numbers → guaranteed
confabulation zone, no curation needed.
- **Why control:** proves the mechanism end-to-end even if the fact-bank is fiddly.
- **Caveat to watch:** for arithmetic, surprisal of the answer may leak *some*
  signal (an unsure model may emit higher-entropy digits), so surprisal AUROC
  might not be as flat here as on facts. Report it; if surprisal partly works on
  arithmetic, that's informative, not a failure — the *factual* substrate is the
  one where surprisal is meant to be flat.

---

## 4. Calibration — the human-in-the-loop step (do NOT skip, do NOT automate away)

Before any AUROC is trusted, the question pool must be filtered to the
confabulation zone **by looking at raw samples**, per the repo's cardinal rule.

1. Run `sampled_consistency` (k≥6, temp 0.8) on every candidate question.
2. For each, record: modal answer, is-modal-correct (vs external ground truth),
   `distinct_ratio`, `hedge_rate`, and the raw sampled answers.
3. **Read the raw samples.** Classify each question:
   - **Known** (modal correct, low distinct, no hedge) → keep as negatives.
   - **Confabulation** (modal wrong, high distinct, low hedge) → keep as positives.
   - **Refusal** (high hedge) → **discard** (that's F26's regime, not this one).
4. Keep only Known + Confabulation items; the eval needs a healthy count of both
   (aim ≥15 each). If almost everything is Known → questions too easy, go obscurer.
   If almost everything is Refusal → too impossible, make them more plausible.

This filtering *is* the task design, and it is why this is not a cold-agent job:
the confabulation zone is found empirically, by eye, per model.

---

## 5. The eval & pre-registration (`loop/f28_confab.py`)

Detection target: flag questions whose answer the model should **not be trusted
on** (the Confabulation positives) vs the Known negatives.

- **Leg 2 signal:** `distinct_ratio` (and/or `1 − agreement`) from
  `sampled_consistency`. Higher = less trustworthy = flag.
- **Leg 1 baseline:** teacher-forced surprisal of the model's own *modal* answer.
  (Score the answer the model actually produced — a confabulation is self-consistent
  and thus low-surprisal, which is the point.)
- **Reuse:** `consistency.sampled_consistency`, `f22_ensemble.auroc`,
  `base.raw_claim_score` for the surprisal baseline; mirror `f26_sample.py`.

**Pre-registered (write results BEFORE reading them):**
- **H-B-1 (Leg 2 works):** `AUROC(distinct_ratio)` separating Confabulation from
  Known **≥ 0.70**.
- **H-B-2 (Leg 2 is NECESSARY — the load-bearing one):** on the *same* split,
  `AUROC(surprisal-of-answer) ≤ 0.60`. This is the whole point: variance detects
  what the model's own confidence hides. If surprisal *also* separates well, Leg 2
  isn't uniquely needed and the routing-on-quality story weakens — report that
  honestly.
- **H-B-3 (not the refusal regime):** mean `hedge_rate` on the positives is low
  (say < 0.3), confirming these are confabulations, not abstentions. If hedge is
  high, the pool drifted into F26 and the result doesn't count.
- **Multi-seed:** 42/137/2024 (the repo standard — do not repeat F27's single-seed
  slip). Report per-seed and mean.

**Then, and only then, the router payoff:** on a mix of context-derivable spans
(entity_prose) + these confabulations, show `AUROC(routed) > AUROC(leg2-only)` —
because now there ARE Leg-2-blind spans (the context-contradicted ones surprisal
catches and sampling may miss) *and* surprisal-blind spans (these confabulations).
This is the F27 payoff the current benchmark structurally cannot deliver.

---

## 6. Known traps (in addition to the repo-wide ones in `BUILD-HANDOFF.md` §1)

- **Zone drift** is the #1 failure. Re-audit the pool's zone split every time the
  question set or the model changes; an eval whose positives quietly became
  refusals (F26) or knowns (F27) will produce a real-looking but meaningless AUROC.
- **Circular labeling.** Label positives/negatives by **external ground truth**
  (modal answer correct?), never by the detector's own output. Using
  `distinct_ratio` to both define and detect confabulation is circular and will
  manufacture a perfect score.
- **Answer normalization** (`consistency._normalize_answer`) is fragile for
  free-form facts ("Paris" vs "Paris, France" vs "the city of Paris"). Inspect
  matches; loosen to substring/entity-contains where needed, and note it.
- **k must be large enough** for `distinct_ratio` to have resolution — k=6 gives
  granularity 1/6; fine, but don't over-read differences < 1/k.
- **hedge vs confabulation** can co-occur within a pool; keep them as separate
  measured quantities, don't collapse.

---

## 7. Deliverables

- `ood/obscure_facts.py` — the curated bank + a `generate(seed, n)` that yields
  `OODTrace`-compatible items (or a thin question/ground-truth structure the runner
  can consume). Hard-arithmetic control alongside or in `arithmetic.py`.
- `loop/f28_confab.py` — the runner: calibration dump mode + the pre-registered
  eval (H-B-1/2/3, multi-seed) + the router-payoff mix.
- `tests/test_confab.py` — CPU tests (question bank integrity, normalization,
  labeling-by-ground-truth not by detector).
- `FINDINGS.md` F28 — honest write-up, including the zone split of the final pool
  (how many Known / Confabulation / discarded-Refusal), so the result is
  interpretable.

**First step for whoever builds it:** draft ~60 candidate obscure questions, run
the §4 calibration, and *show the reviewer the zone split and a sample of raw
answers* before building the eval. If the pool won't populate the middle zone,
that's the finding to surface first — it would mean Qwen3-1.7B either knows or
refuses these, and we need a different obscurity band (or a bigger model).
