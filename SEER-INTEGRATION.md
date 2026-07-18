# SEER Integration — Retrofit Plan for a Pretrained Model

**Status:** working integration doc, 2026-07-17. This file supersedes the
muddled parts of the `seer` repo for now; `seer` stays as the design
archive until we fold the conclusions back. New LLM-scale work lives HERE
(a future `llm/` subpackage), self-contained as always.

## 0. The direct answer

**Yes: everything the evidence says matters is retrofittable onto a
pretrained Qwen3 or Gemma 4 checkpoint. The only component that genuinely
requires from-scratch pretraining is SPS — and our own experiments
(sps-blindspot, terminal conclusion) showed SPS is orthogonal to
self-correction. It is an efficiency/state-fidelity mechanism, not an
awareness mechanism.** So the muddle dissolves: drop SPS from the awareness
plan entirely. It re-enters only if we ever pretrain for other reasons.

What seer actually needs, in the vocabulary of this project's findings:

| seer pillar | finding | retrofit? | mechanism on a pretrained LLM |
|---|---|---|---|
| Transport (SPS) | orthogonal to blind spot (sps-blindspot H1/H2) | ✗ (pretraining-time gradient routing) | skip |
| Inspection (energy/self-certainty) | zero-shot surprisal detection, R3 | ✓ zero training | logp of own sampled token under own state + retrospective re-scoring |
| Inspection (J-space probe) | AUROC .96 on confident errors, but L1 type-bound | ✓ tiny training | linear/LoRA head, confhead joint-objective recipe — SECONDARY to the generative signal |
| Admission: robustness | rehearsal dominates repair, R5 | ✓ LoRA fine-tune | self-output-span dropout during fine-tuning |
| Admission: action gate | thresholds belong on actions, R5/directives | ✓ inference loop | flag → excise from context/KV (true removal, R4) → regenerate; escalate to efh-core |

## 1. Why the LLM port is EASIER than the toy

The surprisal gate needed a two-pass trick in the toy because emissions
were teacher-forced. An autoregressive LLM computes the ingredient
natively: at the moment it samples a token, it has that token's log
probability under its own predictive state. Provenance is free too — the
runtime always knows which context tokens the model itself generated.
Nothing about Inspection requires touching the weights.

Two operational forms of the signal, both label-free:

- **Emission-time surprisal** (online, free): -logp of each sampled token
  / span as it is produced. High = the sampler forced an improbable
  continuation or the distribution was flat. This is the cheap stream.
- **Retrospective surprisal** (the ghost-catcher, R3's true analog):
  re-score previously self-generated spans under the CURRENT state — one
  teacher-forced forward pass over the existing context, reading logp of
  each self-token given everything before it. This catches "I now know
  better": content that was plausible when written but contradicts what
  the accumulated evidence now implies. In the toy this detected every
  corruption type zero-shot at AUROC .93-1.00.

Calibration: per-domain affine on the span-aggregated signal (mean or min
logp over the span), with the slope sign-anchored (softplus — F15b: idle
calibrations drift, one basin inverted). Aggregate at the span level, not
the token level: single-token surprisal on natural text is noisy (rare
words are surprising and fine); the toy's per-emission unit corresponds to
a claim-sized span, not a subword.

## 2. The retrofit stack

Layer 0 — **Provenance bookkeeping** (runtime, no weights). Track
self/external spans in every context. Trivial in any agent loop; the point
of R2 is that discarding this bit is a choice, and the wrong one.

Layer 1 — **Telemetry** (no weights). Emission-time + retrospective
surprisal over self-spans, calibrated per §1. Deliverable: a per-span
trust score for everything the model has said in-context.

Layer 2 — **Self-probe head** (hours of LoRA). The sibling's validated
confidence head (joint objective with a task anchor — confidence-only
fine-tuning catastrophically forgets) reading residual state at
claim-commit positions. Kept SECONDARY: the L1 lesson and R2 both say
discriminatively-trained signals are type/domain-bound; the probe
complements the generative signal in-domain, never replaces it.

Layer 3 — **Hole rehearsal** (the one real fine-tune; QLoRA-sized).
R5's dominant repair factor, ported: during fine-tuning, randomly mask or
excise a fraction of the model's own prior-turn spans (attention-mask them
out or literally remove them) while supervising the same targets — the
LLM analog of emission dropout. Teaches operate-across-holes so that
Layer 4's excisions land in-distribution instead of OOD. Data: any
multi-turn/agentic traces where the model's own prior outputs matter;
even self-generated rollouts suffice (the toy needed zero corruptions).

Layer 4 — **Admission loop** (inference control, no weights). When
telemetry flags a self-span: excise it (delete from context / evict from
KV cache — TRUE removal; R4 says biases and soft discounts lose to copy
heads) and REGENERATE from the flag point, or abstain, or escalate
formalizable claims to efh-core's verification gate. R5's monitor-first
lesson applies: thresholds live here, where a false positive costs one
extra generation, not a belief.

## 3. Model targets (checked 2026-07-17)

Gemma 4 shipped 2026-04-02 under Apache 2.0: E2B / E4B
(effective-parameter, laptop-class), 26B-A4B MoE, 31B dense, 12B unified
multimodal (June). Qwen3 small checkpoints (0.6B / 1.7B / 4B) remain the
workhorses and are what `sparc-falsification` already ran on.

On the 3060 (12GB, fan protocol in force):
- **Phase experiments: Qwen3-0.6B / 1.7B** — full-precision inference and
  LoRA both comfortable; fastest iteration; sibling tooling exists.
- **Headline model: Gemma 4 E4B or Qwen3-4B with QLoRA** — 4-bit base +
  LoRA adapters fits; batch small, gradient-checkpoint.
- 12B+: rent an A100 for the one confirmation run, as with muP plans.

## 4. Phased plan (each phase falsifiable, 3060-sized)

**Phase 0 — Does detection port? (no training, ~a day).** Build the
telemetry harness on Qwen3-1.7B. Take entity-tracking-style multi-turn
traces (reuse the task generator rendered to natural text), inject the
three corruption types into the model's OWN prior outputs, measure
retrospective-surprisal AUROC per span. This is R3's port and the L1
ladder's first rung. Prediction: AUROC ≥ .9 zero-shot. If this fails,
stop and understand why before any training.

**Phase 1 — Does the L1 curse stay broken? (no training).** Same
telemetry on OUT-OF-DOMAIN material (code traces, arithmetic, factual QA
with planted self-errors). The token probe died here at ~.50; the
generative signal's entire value proposition is surviving this. Also
calibrate: reliability curves per domain, one affine each.

**Phase 2 — Hole-rehearsal QLoRA (days).** Layer 3 on Qwen3-1.7B (then
E4B). Measure the toy's trio at LLM scale: does rehearsal cut the
behavioral cost of planted self-errors (bsi analog) without competence
tax, does it beat corrupted-data exposure on held-out corruption types,
and does the trained model — like R5 — shift toward re-derivation?

**Phase 3 — Close the loop (inference engineering).** Layer 4 wired:
flag → excise → regenerate; selective abstention curves (the sibling got
99.7% accuracy at 80% coverage from abstention alone); efh-core
escalation for formalizable claims. Endpoint demo: the model catches a
planted false statement in its own earlier turn, excises it, and
recovers the correct answer — the toy's closed loop, at 1.7B, on text.

## 5. What this replaces in `seer`

seer's Path A ("augment a capable model") is CONFIRMED as the route, with
two corrections from evidence: the energy channel's first implementation
is logprob-based self-consistency (not a new head — Layer 1 before Layer
2), and "energy descent correction" is realized as excise-and-regenerate
(Layer 4) rather than latent-space gradient steps, until Phase 3 data says
otherwise. seer's Path B (from-scratch LANG-JEPA) is not required for any
currently-evidenced capability; park it. SPS: park (see §0). The
three-pillar framing survives intact — Transport was just never about
awareness.
