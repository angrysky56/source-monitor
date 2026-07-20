# Phase 3 — Close the loop on FREE generation

Phases 0–2 validated the pieces separately, all under teacher-forcing or
candidate-ranking:

- **Detection** works zero-shot, scales, and ports OOD within a known boundary
  (F16–F19d).
- **Repair is excision, not training**: attention-masking a flagged span at
  inference restores the answer (base/drop .78 → ~1.00, F20e), while
  hole-rehearsal training added nothing on a short-ξ task.

Phase 3 is the endpoint demo: run the whole loop on **actual generation** and show
the model talk its way out of its own planted error, unassisted.

## Core question

> With no teacher forcing and no answer candidates supplied, can the runtime
> **detect** a false statement in the model's own earlier turn, **excise** it, and
> **regenerate** a correct answer — and does that loop cost anything on clean
> traces?

## The loop (all runtime; no weights)

For a conversation whose earlier assistant turn contains a false claim:

1. **Build context** — turns through the final user query, with
   `add_generation_prompt=True`.
2. **Score self-spans** — retrospective surprisal over every assistant emission in
   context (the validated detector; value-only aggregation for entity claims).
3. **Flag** — within-trace z-score: flag the highest-scoring span if
   `z > k` (default k=1.5). Self-contained: no external calibration set needed,
   and it adapts per conversation. (Percentile calibration on clean traces is
   available as an alternative mode.)
4. **Excise** — set `attention_mask = 0` over the flagged span's CONTENT tokens.
   This is true removal (toy F12: soft discounts lose to copy heads) and is
   exactly the operation validated in F20e.
5. **Regenerate** — `model.generate(...)` with the holed attention mask, so the
   decoder cannot attend to the excised claim.
6. **Grade / abstain** — parse the generated answer against the trace's known
   candidate values; abstain if no candidate is expressed.

Factual/high-prior-ambiguity claims are routed to external verification rather
than surprisal (F19d) — stubbed here as a hook, since the demo domain is
context-derivable.

## Conditions compared

| condition | excision | purpose |
|---|---|---|
| `monitor_off` | none | baseline: the lie stays visible; expect derailment |
| `monitor_on` | detect → excise flagged span | the full loop |
| `oracle_excise` | excise the KNOWN planted span | ceiling: perfect detection |

`oracle_excise` separates *detector* error from *excision* efficacy — if
`monitor_on` < `oracle_excise`, the gap is detection; if `oracle_excise` itself is
low, excision/regeneration is the problem.

## Metrics

On planted-lie traces:
- **answer accuracy** per condition (free-text, graded against candidates)
- **flag hit rate** — flagged span == the planted lie's span
- **false-excision rate** — a genuine span was excised instead

On CLEAN traces (no lie) — the cost side:
- **false-positive excision rate** (loop excises when nothing is wrong)
- **accuracy preserved** vs monitor_off (the loop must not damage clean behavior)
- **abstain rate**

## Pre-registered predictions & gate

- **P-3.1 (the loop repairs):** `monitor_on` accuracy exceeds `monitor_off` by
  ≥ .15 on planted-lie traces.
- **P-3.2 (detection is the right trigger):** flag hit rate ≥ .80, and
  `monitor_on` within .05 of `oracle_excise` (detector is not the bottleneck).
- **P-3.3 (cheap when idle):** on clean traces, false-positive excision ≤ .10 and
  accuracy drops ≤ .02 vs monitor_off.
- **PASS:** P-3.1 and P-3.3 → the source-monitor loop is a working
  inference-time artifact; the architecture concept is demonstrated end to end.
- **FAIL P-3.1 with high `oracle_excise`:** detection thresholding is the weak
  link → tune k / use the calibrated affine (Phase 1 calibration), not the
  mechanism.

## Files

| File | Type | Purpose |
|---|---|---|
| `src/source_monitor/llm/loop/__init__.py` | NEW | subpackage |
| `src/source_monitor/llm/loop/config.py` | NEW | Phase3Config (k, max_new_tokens, conditions) |
| `src/source_monitor/llm/loop/monitor.py` | NEW | score → flag → excise → generate → grade |
| `src/source_monitor/llm/loop/phase3.py` | NEW | runner: conditions × seeds → jsonl |
| `tests/test_loop.py` | NEW | flagging, holed-mask, grading (CPU mock) |

## Execution order

1. `config.py`, `monitor.py` (+ CPU tests with a mock model/tokenizer).
2. `phase3.py` runner.
3. Full pytest on CPU.
4. GPU endpoint run (Qwen3-1.7B, then 4B) on Ty's go-ahead — generation is
   cheap relative to the contrastive sweeps.

## Verification

- Holed mask: zeros cover exactly the flagged span's content tokens, nothing else.
- Flagging: on a synthetic score vector the z-rule picks the intended span; no
  flag when scores are flat.
- Grading: recognizes each candidate value and the negation surface; returns
  abstain when the output expresses none.
- No repair claim until ≥3 seeds, and `oracle_excise` is always reported beside
  `monitor_on` so detector error is never hidden inside the headline.
