# Phase 1 — Does the detection signal survive Out-Of-Domain?

Phase 0 established that retrospective self-consistency (surprisal / contrastive)
detects a model's own planted errors zero-shot on a synthetic entity-tracking
task: containers from 1.7B, absence from ~4B (contrastive + mean-agg). Phase 1
asks the question that killed the sibling project's L1 discriminative probe:

## Core question

> A discriminatively-trained "was this emission corrupted?" probe collapses to
> AUROC ~.50 out-of-domain (the L1 curse; R2/F4). The generative
> self-consistency signal is label-free and type-agnostic *by construction*.
> **Does it keep working when the domain changes entirely** — arithmetic, world
> knowledge, code — with no training, no task-specific tuning?

If yes, that is the whole value proposition: an inspection signal that ports
across domains, which supervised probes cannot. If it collapses too, the
generative bet is in trouble and we learn where.

## Design principle: a near→far OOD gradient

To make the verdict conclusive rather than anecdotal, the domains span increasing
distance from the Phase 0 training-analog task, and increasing difficulty for the
scorer:

| domain | distance | claim | candidates | scorers |
|---|---|---|---|---|
| `entity_prose` | near (same task, free prose, no rigid template) | object location | enumerable (containers ∪ none) | raw + contrastive |
| `arithmetic` | mid (synthetic reasoning) | running/total value | enumerable (near-miss numbers ∪ "undefined") | raw + contrastive |
| `factual_qa` | far (real-world knowledge) | entity answer | enumerable (distractors ∪ "unknown") | raw + contrastive |
| `code_trace` | far (agent-like, free-text) | variable value / output | NOT enumerable | raw only |

`entity_prose` is the bridge: if the signal breaks just from dropping the rigid
template, we learn that before blaming true OOD. `code_trace` is the stress test:
free-text, so contrastive is unavailable and raw surprisal must carry it alone —
the condition closest to real agent monitoring.

## Carry-forward from Phase 0 (must be honored)

1. **Matched-surface stratification (A2).** Compare corrupted claims only against
   genuine claims of the same surface type. Report pooled AND matched; matched is
   the honest number. The Phase 0 corruption confound (drawing corruptions from a
   larger surface pool than appears in-context) must NOT recur — corruptions are
   drawn from the trace's own in-context value set.
2. **Absence/negation is the known weak axis.** Every domain includes a
   negative-claim variant ("no solution" / "undefined" / "not found" / "none"),
   and we report it separately. Phase 0 showed raw surprisal underperforms on
   absence and needs contrastive + sentence-level (mean) aggregation. Phase 1
   tests whether that pattern generalizes OOD.
3. **Aggregation is a reported ablation (A4).** mean, max, slot/value-only —
   report all three. Expect mean to dominate for negation, value-only for
   positive point-claims.
4. **Contrastive where constructible, raw everywhere.** Carry both scorers
   (SEER-INTEGRATION §4). Free-text domains (code) get raw only — a predicted,
   not surprising, disadvantage on negation there.
5. **Multi-seed discipline.** ≥3 task-RNG seeds, mean ± std; nothing is real
   until it survives seeds (project standing note).

## Trace / claim abstraction (domain-agnostic)

Reuse `task_render.Turn` (role, content, is_self, step_index, is_corrupted,
claim_surface, location_text) and `provenance.tokenize_with_provenance` +
`telemetry.retrospective_surprisal` unchanged. New, in `llm/ood/base.py`:

```python
@dataclass
class OODClaim:
    turn_index: int                 # assistant turn holding the corruptible claim
    surface_type: str               # "value" | "negation" | ... (matched-surface)
    candidate_contents: list[str]   # full assistant-turn contents, one per candidate
    emitted_index: int              # index of the genuine (clean) candidate
    correct_index: int              # index of the ground-truth-correct candidate
    #   candidate_contents=None  -> free-text claim: raw scoring only

@dataclass
class OODTrace:
    domain: str
    turns: list[Turn]
    claim: OODClaim
    def as_trace(self) -> Trace: ...  # wrap for provenance (task=None)
```

Candidates carry the FULL turn content (not just the value), so each domain owns
any structural variation (e.g. "The total is 42." vs "There is no valid total.")
— this is the generalization of Phase 0's container/"nowhere" special-casing and
avoids the substitution artifacts that a naive value-swap would introduce.

Generalized contrastive scorer (`ood/base.py`, reuses telemetry helpers
`render_chatml`, `_encode_candidate`, `_mean_lp`, `logsumexp`):

```python
def contrastive_claim_scores(model, tokenizer, ood_trace, device) -> ClaimScore:
    # renormalize the emitted claim against candidate_contents at the claim slot,
    # returning mean/max/value-only aggregations (mirrors Phase 0 contrastive).
```

## Per-domain generators (`llm/ood/*.py`, deterministic, seeded)

Each exposes `generate(seed, n, ...) -> list[OODTrace]` and injectors returning a
corrupted `OODTrace` (positive-claim corruption + a negation corruption), plus the
in-context candidate set.

- **entity_prose** — reuse `source_monitor.task.generate_task`; render each op to
  varied natural prose (templated paraphrase pool, not the fixed Phase 0 string).
  Corruptions: wrong container (misloc-analog), "isn't anywhere" (phantom-analog).
- **arithmetic** — chained ops from a seed value (+,−,×), assistant reports the
  running total each step; correct via Python. Corruptions: near-miss total
  (±1/±10/digit-swap, in-context plausible), and a negation ("no whole-number
  result" on a rigged non-integer division). Candidates: the true total + the
  near-misses + "undefined".
- **factual_qa** — a curated in-repo fact bank (capitals, elements, authors,
  etc.; ~80 items, no network → deterministic). Multi-turn: a few facts stated,
  then the queried one. Corruptions: distractor entity from the same category,
  and "unknown"/"no record". Candidates: correct + same-category distractors +
  "unknown".
- **code_trace** — generate straight-line code (assignments + int ops); assistant
  narrates a variable's value after some steps; correct via Python exec of the
  generated snippet in a sandboxed namespace. Corruption: wrong value; negation:
  claims a defined variable is "undefined"/"not set". Free-text → `candidate_contents=None`.

## Metrics & calibration (`llm/ood/phase1.py`)

Per domain × scorer (raw, contrastive) × aggregation (mean/max/value):
- pooled AUROC and **matched-surface AUROC** (positive-vs-positive, negation-vs-negation);
- paired delta (corrupted − clean twin), same claim slot;
- three genuine populations (clean / pre-claim / post-claim) as in A3;
- **per-domain calibration**: reliability curve (binned genuine-vs-corrupt score)
  and a single fitted affine (slope sign-anchored via softplus, per F15b) mapping
  score → P(corrupt); report ECE.

Output: `results/llm_phase1_results.jsonl`, one flat record per
(domain, model, scorer, seed), with transformers/model-revision/env provenance (A5).
Reuse `scripts/aggregate_phase0.py` pattern in `scripts/aggregate_phase1.py`.

## Pre-registered predictions & gate

- **P-1.1 (transfer holds):** matched-surface AUROC ≥ .85 for positive claims on
  every enumerable-candidate domain (entity_prose, arithmetic, factual_qa) at 4B
  under the best of {raw, contrastive}. This is the L1-curse-stays-broken result.
- **P-1.2 (contrastive still rescues negation):** on negation claims, contrastive
  + mean beats raw by ≥ .05 on arithmetic & factual_qa (the Phase 0 pattern OOD).
- **P-1.3 (free-text floor):** code_trace raw matched-surface AUROC ≥ .70
  (well above the .50 the discriminative probe would give) — graceful, not
  collapsed.
- **P-1.4 (calibration):** per-domain affine yields ECE ≤ .10; a SINGLE global
  affine does materially worse (justifying per-domain calibration, SEER §1).
- **PASS:** P-1.1 and P-1.3 hold → the generative signal ports OOD; proceed to
  Phase 2 (hole-rehearsal QLoRA). **FAIL P-1.1:** the signal is domain-bound after
  all — characterize which domains and why before any training (this would be a
  major negative result and must be reported, not smoothed).

## File structure

| File | Type | Purpose |
|---|---|---|
| `src/source_monitor/llm/ood/__init__.py` | NEW | subpackage |
| `src/source_monitor/llm/ood/base.py` | NEW | OODTrace/OODClaim + generalized contrastive scorer |
| `src/source_monitor/llm/ood/entity_prose.py` | NEW | near-OOD bridge generator |
| `src/source_monitor/llm/ood/arithmetic.py` | NEW | arithmetic generator + corruptions |
| `src/source_monitor/llm/ood/factual_qa.py` | NEW | fact-bank QA generator |
| `src/source_monitor/llm/ood/code_trace.py` | NEW | free-text code-state generator |
| `src/source_monitor/llm/ood/phase1.py` | NEW | runner + calibration |
| `tests/test_ood_*.py` | NEW | per-domain + scorer tests |
| `scripts/aggregate_phase1.py` | NEW | reporting + gate eval |

## Execution order

1. `base.py` (+ test with mock) — the abstraction and generalized scorer.
2. `entity_prose.py` (+ test) — the bridge; validates the abstraction end-to-end.
3. `arithmetic.py`, `factual_qa.py`, `code_trace.py` (+ tests).
4. `phase1.py` runner + `aggregate_phase1.py`.
5. Full pytest on CPU (generators + scorer use mocks/tiny tensors — no GPU).
6. GPU sweep (1.7B + 4B) is Phase 1 *execution*, run only on Ty's go-ahead
   (fan protocol; contrastive is slow — budget like Phase 0).

## Verification

- Unit: determinism (same seed → same traces), corruption validity (only the
  claim turn changes; negation variants say the negation surface), candidate
  in-context invariant (no out-of-context distractors → no Phase-0 confound),
  ground-truth correctness (arithmetic/code checked against Python).
- Scorer: mock-model equivalence of the generalized contrastive path to a hand
  computation on a tiny example; span alignment on the char tokenizer.
- No claim of transfer is made until the GPU sweep runs ≥3 seeds on 1.7B and 4B.
