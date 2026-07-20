# Phase 2 — Hole-Rehearsal LoRA (the repair leg, at LLM scale)

Phases 0–1 established the **inspection** channel: retrospective self-consistency
detects a model's own planted errors zero-shot, ports OOD for context-pinned
low-ambiguity claims, and its boundary is now characterized. Detection, however,
is not repair (toy F9): evicting/flagging a bad self-emission does not by itself
rescue the next step. The toy's decisive repair factor was **rehearsal** — training
under real self-report blackouts (emission dropout at -1e9) built a re-derivation
circuit and distrust-by-default (F13b/F14). Phase 2 ports that to Qwen3.

## Core question

> If we fine-tune the model to operate when a fraction of its OWN prior emissions
> are excised (holes) — with NO corruption ever shown in training — does it become
> behaviorally robust to planted self-errors at inference (re-derive the right
> answer despite a lie in context), WITHOUT a competence tax, and does that
> robustness transfer to corruption types it never saw? And does it beat simply
> training on corrupted data (which the toy showed is corruption-type-bound, F4)?

## Mechanism: holes = hard attention masks on self-emission spans

During fine-tuning, for each training trace we supervise the assistant emissions
(SFT loss on assistant content tokens only), but randomly select a fraction
`p_hole` of the EARLIER assistant emission spans and **mask them out of attention**
(attention_mask = 0 over those tokens) for the whole forward pass. The model must
therefore predict later emissions without attending to (some of) its own prior
emissions — forcing re-derivation from the user turns (external evidence).

Why hard masking, not a soft discount: toy F12 pinned that an additive -30 bias
ATTENUATES rather than evicts (trained copy heads out-shout bounded biases); only
-1e9 / true removal rehearses genuinely. attention_mask=0 is the LLM analog of
-1e9 (the token contributes nothing to any query). This is emission dropout, R5's
mechanism, ported.

Two implementation guards (carried from Phase 0 A1):
- Mask CONTENT tokens of the chosen assistant spans only — never role markers /
  template boilerplate (those must stay visible or the chat structure breaks).
- Holes are applied to spans strictly BEFORE the span currently supervised (a
  hole never masks the label span itself).

## Arms (reproduce the toy's decomposition at LLM scale)

All LoRA on Qwen3-1.7B, bf16 (no 4-bit needed at 1.7B; QLoRA/bitsandbytes only
enters for 4B/E4B later). Identical data, seeds, steps across arms:

1. **base** — LoRA on clean traces, no holes, no corruption. Control for "does
   any fine-tuning on this task shift behavior."
2. **drop** — LoRA on clean traces + hole rehearsal (`p_hole>0`), ZERO corruption
   exposure. The R5 arm.
3. **corrupt-exposure** — LoRA on traces containing planted corruptions with the
   CORRECT targets supervised (learn to answer through a lie). The F4 comparison:
   expected to fix the trained corruption type but not transfer.

## Training data

Clean multi-turn traces from the existing generators (entity_prose primary — the
toy-comparable task; arithmetic secondary), where the model's own prior emissions
carry state. No corruption in `base`/`drop`. `corrupt-exposure` injects the SAME
in-universe corruptions used in Phase 0/1 (ghost/misloc-analog) but supervises the
genuine answer. Hold out one corruption type from `corrupt-exposure` training to
measure transfer.

## Evaluation (the bsi analog + competence + transfer)

Reuse Phase 0/1 inference telemetry. Per arm × base-model:

1. **Planted-error recovery (bsi analog).** Take clean eval traces; plant a false
   self-emission at a non-final step k (wrong location / value); teacher-force the
   trace through k; then measure whether the model's emission at the FINAL step is
   CORRECT (re-derived) despite the lie in context. Metric: final-answer accuracy
   under planted error. bsi = (clean final accuracy − planted final accuracy).
   Rehearsal should shrink bsi.
2. **Competence tax.** Final-answer accuracy on CLEAN traces (no planted error),
   base vs each arm. Rehearsal must not degrade it (toy: dropout even helped).
3. **Held-out transfer.** bsi on a corruption type NOT seen by `corrupt-exposure`;
   the drop arm (which saw zero corruptions) should transfer uniformly, corrupt-
   exposure should not (F4).
4. **Detection survives.** Re-run Phase 0 retrospective-surprisal AUROC on the
   LoRA'd model — rehearsal should not damage the free inspection signal.

## Pre-registered predictions & gate

- **P-2.1:** `drop` reduces planted-error bsi vs `base` by a meaningful margin
  (target: bsi roughly halved) with competence tax ≤ 2 points. (toy F14/F13a.)
- **P-2.2:** `drop` transfers to held-out corruption types; `corrupt-exposure`
  is type-bound (helps its trained type, not held-out). (toy F4, the central
  L1-curse result — now at LLM scale.)
- **P-2.3:** detection AUROC (Phase 0 telemetry) is preserved (≥ .95 of base) on
  the rehearsed model.
- **PASS:** P-2.1 and P-2.2 → the repair leg ports; rehearsal (not corrupted-data
  exposure) is the transferable robustness mechanism, confirming the toy's R5 at
  LLM scale → proceed to Phase 3 (close the loop: flag→excise→regenerate).
- **FAIL P-2.1:** hole rehearsal alone does not repair at 1.7B — characterize
  (more steps? p_hole? scale to 4B?) before Phase 3. Multi-seed before any claim.

## Dependencies

Only **`peft`** is new (LoRA). bf16 LoRA on 1.7B fits the 12 GB 3060 with room;
`bitsandbytes` (4-bit) deferred to the 4B/E4B confirmation. `trl`/`datasets` not
used — a small custom loop keeps full control of the hole-masking collator, which
an off-the-shelf SFT trainer cannot express. Add via `uv add peft`.

## File structure

| File | Type | Purpose |
|---|---|---|
| `src/source_monitor/llm/train/__init__.py` | NEW | subpackage |
| `src/source_monitor/llm/train/config.py` | NEW | Phase2Config (arms, p_hole, LoRA cfg, steps) |
| `src/source_monitor/llm/train/dataset.py` | NEW | clean traces → SFT examples (assistant-only labels) |
| `src/source_monitor/llm/train/hole_collator.py` | NEW | hole masking + corruption-exposure collators |
| `src/source_monitor/llm/train/lora_train.py` | NEW | bf16 LoRA loop (peft) |
| `src/source_monitor/llm/train/eval_repair.py` | NEW | bsi / competence / transfer / detection eval |
| `tests/test_train_holes.py` | NEW | collator + label-masking tests (CPU, no peft) |
| `scripts/aggregate_phase2.py` | NEW | arm comparison + gate |

## Execution order

1. `config.py`, `dataset.py`, `hole_collator.py` (+ tests) — all CPU-testable, no peft.
2. `eval_repair.py` (reuses Phase 0/1 inference; CPU-testable on mock).
3. `lora_train.py` (needs peft) — write, then `uv add peft`, then a tiny smoke train.
4. Full pytest on CPU (no-peft parts).
5. GPU training run (3 arms × 3 seeds, 1.7B) only on Ty's go-ahead (fan protocol;
   LoRA on 1.7B is far cheaper than the contrastive sweeps).

## Verification

- Collator unit tests: masked spans are exactly the chosen assistant CONTENT
  token ranges (no boilerplate), labels are -100 everywhere except supervised
  assistant content, a hole never covers the supervised span, determinism per seed.
- eval_repair on a mock model returns sane bsi/competence structure.
- Detection-survives check reuses the committed Phase 0 telemetry unchanged.
- No repair claim until ≥3 seeds on 1.7B (basin-variance discipline).
