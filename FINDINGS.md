# source-monitor — findings log

## F21 — Phase 3 (closed loop on FREE generation): the loop WORKS; the
## trigger policy does not (2026-07-19; Qwen3-1.7B, n=60 x 3 seeds).

    split     condition        accuracy   excise   flag hit
    planted   monitor_off      .833       .000     --
    planted   monitor_on       .961       .917     .952
    planted   oracle_excise    .978       1.000    1.000
    clean     monitor_off      1.000      .000     --
    clean     monitor_on       .961       .956     .000

**F21a — The endpoint demo works.** On free generation, with no teacher forcing
and no candidates supplied, the loop detects a false statement in the model's own
earlier turn, excises it, and regenerates: .833 -> .961, i.e. the error rate falls
16.7% -> 3.9% (a 77% cut). Detection is a near-perfect TRIGGER: flag hit .952 and
monitor_on lands within .017 of oracle_excise -> **P-3.2 PASS**; the detector is
not the bottleneck.

**F21b — P-3.1's "FAIL" is a BAR error, not a mechanism failure (my
pre-registration was wrong).** oracle_excise = .978 against monitor_off .833, so
the MAXIMUM achievable gain on this task is +.145 — below the +.15 bar I
pre-registered. The bar exceeded the task's ceiling. The loop captured
.128/.145 = 88% of the achievable repair. Report it that way, not as a failure.

**F21c — P-3.3 FAIL is the real, actionable defect: the trigger is
indiscriminate.** On CLEAN traces the monitor excises 95.6% of the time and costs
3.9 points of accuracy (1.000 -> .961). Cause: the flag rule is a purely RELATIVE
within-trace z-score. With only ~4-5 spans per trace the maximum is almost always
>= 1.5 sigma (for n=5 the largest possible z is ~2.0), so it fires whether or not
anything is wrong. FIX: require an ABSOLUTE floor — the Phase 1 per-domain
calibrated affine — alone or ANDed with the relative z. Planted lies carry
genuinely high absolute surprisal; clean spans do not.

**F21d — Two harness bugs, both surfaced only because this is the first phase
that GENERATES.** (1) Qwen3's <think> scaffolding was never actually disabled:
`enable_thinking=False` was recorded as metadata but never affected generation, so
the model spent its whole budget reasoning and emitted no answer (all conditions
abstain=1.000, an uninterpretable first run). Fixed by prefilling the empty think
block, plus stripping any leaked think text BEFORE grading — the reasoning names
locations, so an ungated grader would score the model's deliberation instead of
its answer. (2) A literal negation cue list mis-scored correct answers
("is not CURRENTLY anywhere") as abstains; replaced with a pattern. Both tested.

**Next:** (1) absolute/calibrated threshold, then re-run (cheap — this is a
policy change, not a mechanism change); (2) long-ξ task; (3) 4B.

## F20 — Phase 2 (hole-rehearsal LoRA): holes alone do NOT repair visible
## lies (P-2.1 FAIL) — a faithful reproduction of toy F11, with a task-ξ
## confound (2026-07-19; Qwen3-1.7B, 3 arms x 3 seeds).

    arm         competence     planted_acc    bsi            detect_auroc
    base_noft   .877           .680           .197           .979
    base(LoRA)  1.000          .787           .213           1.000
    drop(holes) 1.000          .783           .217           .999
    corrupt     1.000          1.000          .000           .999

**F20a — Ordinary fine-tuning fixes competence, not the blind spot.** base
(LoRA, no holes) drives competence .88->1.0 and detection ->1.0, but bsi stays
~.21 (planted self-lie still derails the final answer ~20% of the time).

**F20b — Hole-rehearsal alone does NOT reduce bsi (drop .217 ~ base .213).**
Reproduces toy F11: rehearsing ABSENCE (attention-masked emissions) does not
create distrust of a VISIBLE, plausible lie (absent != visibly-false). In the toy
(F13b) repair required detection -> true EVICTION (turn the lie into a hole) ->
rehearsed re-derivation. The Phase 2 eval leaves the lie visibly in context with
no excision, so holes have nothing to act on. Detection survives LoRA (P-2.3 PASS).

**F20c — corrupt bsi=.000 is IN-DISTRIBUTION memorization, not transfer.** The
corrupt arm trained and was evaluated on the SAME mid-ack lie; perfect repair is
expected and uninformative. Real test (P-2.2 held-out corruption type) not yet
implemented; toy F4 predicts corrupt-exposure is type-bound.

**F20d — Task-ξ confound (Haggi-Mani & Rish 2026, arXiv:2607.15449, "Relevant and
Irrelevant: an RG analysis of attention").** Their result: attention to prior
tokens is "irrelevant" for short correlation-length (ξ) data. entity_prose's
answer is derivable from the LAST user turn (short ξ), so masking PRIOR emissions
costs nothing -> hole-rehearsal has no re-derivation pressure. Prediction:
hole-rehearsal should only bite on a long-ξ task (arithmetic running total;
"where was X N steps ago").

**F20e — EXCISION repairs the blind spot; hole-rehearsal adds nothing on this
(short-ξ) task (excised-lie eval, saved adapters, 3 seeds).** Planted-error
final-answer accuracy, lie VISIBLE -> EXCISED (attention-masked at inference):
base_noft .68->.86 (+.18), base(LoRA) .79->.998 (+.21), drop .78->1.000 (+.22).
drop vs base under excision: 1.000 vs .998 (Δ+.002, negligible). Conclusion: the
repair leg is EXCISE-AND-REGENERATE at inference (SEER Layer 4), NOT the
hole-rehearsal training; what training buys is COMPETENCE (base LoRA .88->1.0),
which makes excision-recovery near-perfect. drop~base confirms F20d — hole
rehearsal's distinct value needs a long-ξ task. This closes the loop the toy
predicted (monitor-first; true removal, not soft discount; robustness structural)
at LLM scale: detect (Phase 0/1) + excise-and-regenerate (here) = repair.

**Next (owed):** (1) re-run drop on a long-ξ task (arithmetic running total) to
test hole-rehearsal's reserved role. (2) Phase 3: wire the closed loop on free
generation (detect -> excise/KV-evict -> regenerate; abstain; escalate factual
claims to external verification per F19d).

## F19 — Phase 1 (OOD transfer): the signal ports for CONTEXT-DERIVABLE
## claims (3/4 domains), fails on pure factual RECALL — and recall
## detection INVERTS with scale (2026-07-19; n=200 x 3 seeds, 1.7B + 4B).

Matched-surface AUROC, best-of-{raw,contrastive}, mean aggregation:

    domain         value (1.7B/4B)   negation (1.7B/4B)   scorers
    entity_prose   .998 / 1.000      .976 / 1.000         raw+contrastive
    arithmetic     1.000 / 1.000     1.000 / 1.000        raw+contrastive
    code_trace     .917 / .852       .901 / .970          raw only (free-text)
    factual_qa     .565 / .538       .643 / .382          raw+contrastive

**F19a — Transfer holds wherever the answer is derivable from the trace's own
context.** entity_prose (format change only), arithmetic (new reasoning domain),
and code_trace (free-text, agent-like) all detect planted self-errors at .85-1.00
matched — decisively above the ~.50 a discriminative L1 probe gives OOD (R2/F4).
The Phase 0 lessons replicate OOD: contrastive+mean rescues negation
(entity_prose neg .855 -> .976 at 1.7B), and detection scales up with size.
P-1.3 (free-text floor) passes at .85.

**F19b — factual_qa FAILS at chance, and negation detection INVERTS with scale.**
Pure world-knowledge recall sits at AUROC ~.50 at both scales. Worse, the 4B raw
negation AUROC is .000 (perfectly inverted, +-.000 over seeds): for unanswerable
questions the model finds the correct "no reliable record" MORE surprising than a
confident fabricated answer — every time. Mechanism: retrospective surprisal
measures the model's own ANSWER PRIOR, not truth. When truth is not in-context,
surprisal reflects belief; and larger models confabulate more fluently, so
"I don't know" looks anomalous. This is the opposite of Phase 0's absence scaling
(which was context-derivable).

**F19c — Verdict: PASS-with-scope, not FAIL.** The signal is claim-TYPE-bound
(derivable-inconsistency vs factual-recall), NOT domain-bound — it ports across
formats, reasoning, and free-text agent traces. This matches the architecture:
telemetry catches derivable inconsistency; factual/recall claims must route to a
different actor (verification / retrieval / efh-core escalation, SEER Layer 4),
never to surprisal. Owed next: a confirmatory in-context-factual variant (state
the fact earlier in the trace) — it should jump to high AUROC, proving the
boundary is derivability, not the factual domain per se. Then Phase 2
(hole-rehearsal QLoRA).

## F19d — CONFIRMATORY TEST FALSIFIED the naive "derivability" claim; the real
## boundary is competing-PRIOR ambiguity (2026-07-19). Supersedes F19c's guess.

F19c predicted an in-context-factual variant (answer stated in the trace) would
jump to high AUROC. It did NOT: factual_grounded matched value AUROC .55-.58 at
1.7B (3 seeds) and .58 at 4B — chance, no better than ungrounded recall. A
copy-diagnostic (scripts/diag_factual_copy.py; 1.7B, value-token raw AUROC,
corrupt vs genuine) explains why:

    A nonsense-copy      0.792   in-context NOVEL fact ("secret code is Zebra")
    B realfact-plain     0.578   natural grounding of a real fact
    C realfact-meta      0.578   generator's phrasing (== B, so not a phrasing artifact)
    D realfact-none      0.484   pure recall

In-context COPYING works (nonsense .79 >> .5), ruling out a scoring bug and the
"setup can't use context" hypothesis. But for REAL facts, grounding does not
help: the model's strong prior over plausible answer tokens swamps the in-context
signal — genuine "Paris" and plausible-wrong "Lyon" carry comparable prior mass
(mean surprisal 21 vs 24) so surprisal can't separate them; a nonsense answer has
no competing prior (14 vs 18) so it can.

REVISED mechanism (supersedes F19c): retrospective surprisal detects
self-inconsistency only when the answer is BOTH pinned by context AND not swamped
by competing parametric priors. Entity/arithmetic/code satisfy both (uniquely
determined, low-ambiguity slots); factual recall fails the second condition and
grounding fixes only the first. Monotone gradient: entity ~1.0 > nonsense-copy
.79 > real-grounded .58 > real-recall .48. Seer implication is sharper than F19c
stated: knowledge/factual claims MUST route to verification/retrieval (Layer 4),
because no amount of in-context grounding makes them surprisal-detectable when
distractors are a-priori plausible. (factual_grounded run: 3 seeds 1.7B + 1 seed
4B, stopped early — the effect was flat and consistent; diagnostic is the decider.)

## F18 — Qwen3-4B closes the scaling question: absence-tracking is
## SCALE-limited, not fundamental (2026-07-19; n=200 x 3 seeds).

Phantom matched-surface AUROC, contrastive, by scale:

    aggregation   0.6B P/H     1.7B P/H     4B P/H
    slot_only     .71/.78      .79/.85      .84/.89     (still < .90)
    mean          .77/.84      .87/.91      .935/.951   (>= .90 BOTH at 4B)

Monotonic in model size and task length on both aggregations. Under the
sentence-level (mean) contrastive metric phantom clears the .90 bar at 1.7B/hard
and at 4B on BOTH configs -> P-0b.1 PASSES at 4B (it FAILED at 1.7B). Under
slot_only it is still climbing (.84/.89 at 4B). Containers stay >= .99 matched at
4B; the ghost pooled << matched anomaly (P-0b.2) persists (a pooled-metric
absence-prior artifact, not a detection failure). Raw phantom also rises with
scale (.83/.89 at 4B), so larger models track absence better even before
renormalization; contrastive+mean adds +.07-.10 on top.

Conclusion: **Phase 0 closes PASS-with-scale-note.** Retrospective
self-consistency detection ports to pretrained LLMs: container claims from 1.7B,
absence claims from ~4B, with absence requiring (a) candidate renormalization and
(b) sentence-level (not slot-only) aggregation. Carry into Phase 1: raw surprisal
underperforms on negative/absence assertions (a predicted OOD weak axis); prefer
contrastive+mean wherever a claim's alternative set is constructible.

## F17 — Phase 0b RESULTS, corrected & confound-controlled (2026-07-19;
## 400 traces x 3 seeds, raw & contrastive, persisted; supersedes the interim
## Phase 0b claims). Reproduce via scripts/aggregate_phase0.py.

**F17a — Two methodology fixes before the numbers are trustworthy.**
(1) *Corruption drew the false container from the global 8-box pool*, not the
3-4 boxes actually named in each trace, so ~50-70% of ghost/misloc corruptions
pointed at a never-seen box — a surface-novelty confound the A2 matched-surface
guard did NOT catch (it matched claim TYPE, not box-seen-ness). Fixed:
corruptions now draw in-universe. Large effect on containers (0.6B misloc slot
paired-delta +28.7 -> +7.1 nats; ghost matched .99 -> .90 at 0.6B); phantom is
untouched (it never picks a box: matched .7133 -> .713). (2) *The contrastive
candidate set was also the global 8-box pool*; restricted to trace containers +
"nowhere" per spec, and the per-candidate Jinja re-render (the "seed-137 hang"
— slowness/thermal, NOT a logic loop) was removed. The prior interim
"partial pass / projected .893 intersects .90" was single-seed, unpersisted,
and pre-confound-fix — only ONE contrastive row had ever reached disk; not
reproducible.

**F17b — Containers port at 1.7B; phantom does not clear the bar, but
contrastive helps at the SENTENCE level.** Matched-surface AUROC:
- containers (1.7B, contrastive): ghost .99, misloc 1.00; contrastive lifts
  ghost matched +.02-.03 over raw. Honest at 0.6B: ghost ~.90-.93 (below the
  old .97 bar only because the novelty inflation is gone).
- phantom is aggregation-dependent:
    slot_only : raw ~ contrastive, ~0 lift   (1.7B: .79 / .85)
    mean      : contrastive +.06-.09, monotonic in scale
                .774 / .837 / .869 / .912   (0.6B-P / 0.6B-H / 1.7B-P / 1.7B-H)
  1.7B/hard mean-agg crosses .90; 1.7B/primary (.869) does not. max: negative
  (uninformative, per F16c). Container matched stays .999-1.00 under mean-agg,
  so the phantom mean-lift is real signal, not global inflation.

**F17c — Gate verdict.** P-0b.1 FAIL (needs phantom matched >= .90 under
contrastive at 1.7B on BOTH configs; primary is .79 slot / .87 mean).
P-0b.2 FAIL (ghost pooled << matched persists at 1.7B, slot: .67-.72 vs .99 —
the anomaly does NOT dissolve). P-0b.3 PASS at 1.7B (containers >= .99).

**F17d — Mechanism & next step.** The absence signal lives in the claim
SENTENCE ("is nowhere" vs "is in X"), not the single slot token: candidate
renormalization recovers most of the phantom gap ONLY under mean aggregation.
F16b's prior-bias story is thus partially right (cancelling priors helps) but
insufficient at <=1.7B, and matched-surface already neutralizes the slot-level
prior (hence slot-only contrastive ~0). Scaling is steep and monotonic, so per
the pre-registered FAIL-P-0b.1 branch: characterize 0.6B->1.7B->4B before any
Phase 2 training. 4B may clear .90 on both configs under mean-agg.

## F16 — Phase 0 (LLM port, Qwen3-0.6B/1.7B): container-claim detection
## ports decisively; phantom fails the amended bar for an identifiable
## reason (2026-07-17; full tables in walkthrough.md)
## [CORRECTED by F17, 2026-07-19: the container matched AUROCs below are
## inflated by the 8-box novelty confound; the phantom conclusion stands.]

**F16a — R3 ports where surface forms match.** Retrospective surprisal on
self-emitted claims, zero-shot, no training: ghost and mislocation
matched-surface AUROC .987-1.000 (1.7B; .969-.998 at 0.6B), paired deltas
huge (misloc slot-only +29 nats). Detection scales UP with model size and
task length. The A2 guard did its job in both directions: containers pass
the artifact test cleanly.

**F16b — The phantom failure and the ghost-pooled anomaly are ONE
finding: absence-assertion prior bias.** Phantom matched-surface AUROC
.778-.838 (< .85 bar, every config); ghost POOLED slot AUROC .835 despite
matched .990. Common cause: the model assigns depressed logp to
"nowhere"-type claims regardless of truth — true absences look surprising
(polluting the genuine pool), false absences can't separate from true
ones. Reporting bias: pretraining text under-asserts absences. The toy
was immune because its priors were task-learned. Phantom's high paired
delta (+8.6) with mediocre matched AUROC is the confirming signature
(both populations shifted, overlap preserved). Per the amended gate:
Phase 0 = container-claims PASS, phantom = INVESTIGATE branch.

**F16c — Aggregation:** max-token surprisal is uninformative in natural
text (AUROC .3-.6); mean and slot-only work. Slot-only is the ceiling for
containers, NOT for phantom (prior bias lives in the slot itself).

**Phase 0b (specified in implementation_plan.md): contrastive slot
scoring.** Replace raw -logp with a candidate-renormalized score: at each
claim slot, score all candidate completions (each container + nowhere)
and use -log[p(emitted)/Σ p(candidates)] — a likelihood-ratio detector
that cancels class priors entirely and isolates the state-contradiction
component. Predictions: phantom matched ≥ .9; ghost pooled ≈ ghost
matched (anomaly dissolves — the internal consistency check); containers
unchanged. If phantom STILL fails with priors cancelled, the model
genuinely cannot track absence at these scales — a real limit, worth
knowing before Phase 1. Also owed from Phase 0: the A3 three-population
split (the cascade population went unreported).

## F15 — Power run, n=9 paired seeds (2026-07-17): the gate's increment is
## real on phantom, suggestive on misloc, null on ghost

base-drop vs surp-drop-hard, paired by seed, exact sign-flip permutation p:

  type     base-drop bsi   surp-drop-hard bsi   increment        p
  ghost    .023 ± .011     .025 ± .015          -.002 ± .019   .848
  misloc   .070 ± .030     .041 ± .026          +.029 ± .044   .121
  phantom  .066 ± .037     .029 ± .033          +.037 ± .035   .023 *
  (phantom d1 +.092, p=.070; pooled bsi increment +.0215)

Verdict: on the trained-difficulty type (ghost) both arms sit at floor
(~.02) — no headroom, null as expected. On the two harder held-out types
the gate roughly HALVES residual bsi, reaching significance on phantom
(p=.023, 8/9 seeds positive) and a same-direction trend on misloc (6/9).
Fair summary: rehearsal carries most of the repair (F14 stands); the
detect-and-evict gate adds a modest, consistent increment precisely where
re-derivation is hardest — established on one of two pre-registered
discriminators, suggestive on the other. A 12-16 seed top-up would settle
misloc; the effect size (~.03 bsi) is worth exactly that much compute and
no more.

**F15b — Telemetry wart: the idle gate's calibration is not sign-anchored.**
Seed 4's gate_auroc came out 0.003-0.004 — PERFECTLY INVERTED ranking
(g_corrupt .989 > g_genuine .973) — while repair stayed mid-pack (rehearsal
doing the work). When equilibria keep gates open, (a, b) feel almost no
gradient and one basin learned an inverted calibration. The zero-shot
monitor is only "free" if its sign is pinned: v5 fix is structural —
parameterize a = softplus(a_raw) (surprisal can only ever CLOSE a gate) —
one line in model.py. IMPLEMENTED same day (init preserves a=1 at start);
runs before this date carry the unanchored calibration.

**Project standing after F15.** Established: the blind spot and its type
gradient; type-boundness of label-supervised admission; zero-shot
label-free detection via self-surprisal; the -inf eviction requirement
against trained copy heads; rehearsal (real holes) as the dominant repair
mechanism; a significant gate increment on phantom. Open: misloc increment
(underpowered), sign-anchored calibration (v5, one line), scale transfer
(everything here is 4.9M/1500 steps).

## Decomposition control: base-drop with real holes, seeds 0/1/2
## (2026-07-17) — F14: rehearsal is the dominant repair factor

base-drop (-1e9 dropout, NO gate, NO provenance) vs surp-drop-hard, bsi
seed-by-seed [s0/s1/s2]:

                ghost              misloc             phantom
  base-drop     .009/.040/.036     .037/.090/.126     .016/.062/.140
  surp-drop-hard .027/.022/.037    .038/.011/.048     .003/.009/.105

**F14.** Rehearsal alone recovers MOST of the repair (floor .10-.17 →
mean .03-.08). The gate adds a suggestive increment on the harder types
(misloc/phantom mean bsi roughly halved) — but seed 0 REVERSES the ordering
and per-seed swings (±.05) are the same size as the claimed effect. At n=3
the gate's incremental behavioral contribution is NOT established; ~6-8
seeds would be needed to settle it. The zero-shot detection channel is
established regardless (per-position AUROC over 1200 pairs, every run).

## Terminal synthesis (v1-v4, one day, 2026-07-17)

1. The blind spot is real and type-graded (F1): d1 .30, bsi .10 (ghost) to
   .17 (plausible lies are costlier than impossible ones).
2. Label-supervised admission = perfect in-type detector, zero transfer
   (F3/F4): discriminative supervision learns corruption SIGNATURES. The
   L1 probe lesson, reproduced at mechanism level.
3. Generative self-consistency (surprisal under own prediction) detects all
   corruption types zero-shot, with no labels and no corruption exposure,
   at no competence cost (F8). THE transferable inspection signal.
4. Bounded attention biases do not evict against trained copy heads whose
   logits exceed the bias (F12): gate to -inf-grade or don't bother. Soft
   "trust discounts" in attention are shouted down. (Three successive
   "deep negative findings" were this one numerical constant.)
5. Repair is carried by REHEARSAL: training under real self-report
   blackouts (emission dropout, -1e9) builds a re-derivation circuit and
   structural distrust-by-default; learned gate calibration then keeps
   gates open — optimization itself chooses monitor-over-evict (F13b).
   Reactive hard eviction without rehearsal partially works but cascades
   false evictions off contaminated beliefs (the d2 dip, F13b).
6. Gate increment over rehearsal alone: suggestive on hard types, not
   established at n=3 (F14).

**Design directive for seer.** Inspection = the surprisal/energy channel,
validated here in miniature (zero-shot, label-free — for an AR LLM the
analog is simply the logprob of the model's own sampled token under its own
predictive state, plus a calibration affine). Admission's repair function
should be REHEARSED RECOMPUTATION (train under masked-self-output
conditions), not in-context masking; gating/thresholds belong on the
ACTION side (abstain, verify via efh-core, recompute) where false positives
are cheap. Epistemic humility as structure, metacognition as telemetry.

**Discipline note.** Every headline in this project flipped at least once
under (a) a faithful instrument, (b) multi-seed replication, or (c) a
missing control. The sibling repo's rules (gate -> faithful lens ->
multi-seed; instrument before conclusion) earned their keep four times in
one day. Basin variance at 1500 steps/4.9M params is the dominant noise
source; nothing at this scale is real until it survives seeds.

## v4 definitive rerun with true eviction (-1e9), seeds 0/1/2 (2026-07-17)
## — P9 PARTIAL PASS, and the mechanism is not the one designed

**F13a — Repair is real.** surp-drop-hard: bsi ghost .027/.022/.037, misloc
.038/.011/.048, phantom .003/.009/.105; d1 .80-.98 (means ~.83-.89 vs floor
.30). surp-clean-hard (no rehearsal): partial — ghost bsi ~.07, but
misloc/phantom bsi still .09-.17; d1 .61-.94. P9's formal thresholds
(d1 ≥ .9, bsi ≤ .03 everywhere) are NOT met — seed variance is large
(s1 nearly perfect: .94/.98/.97 d1, bsi ≤ .022; s2 weakest, competence .977)
— but the blind spot is roughly 3-10x smaller than every previous arm.

**F13b — The plot twist: repair did not come from eviction.** In
surp-drop-hard the gate barely evicts: g_corrupt .49-.97 (OPEN on corrupted
positions!) while ranking stays perfect (auroc .987-1.000). With -1e9
dropout the rehearsal is REAL for the first time (v3's -30 masks leaked, so
the recurrence shortcut survived "rehearsal"): genuinely losing 30% of
emissions forced a first-class re-derivation circuit and globally reduced
emission-reliance — and the learned calibration (a,b get task-loss gradient)
then chose to keep gates open because eviction was no longer worth its
false-positive cost. The system converged on DISTRUST-BY-DEFAULT plus
detection-as-telemetry, not detect-and-evict. Meanwhile surp-clean-hard
(gates DO slam, g_corrupt ~0) shows eviction-only gives partial repair and a
NEW artifact: a d2 dip (.68-.85) from CASCADE evictions — pass-1 beliefs are
ghost-contaminated at k+1, so the TRUE emission k+1 looks surprising and
gets evicted too; pass-1 re-derives by k+2 and the cascade self-limits.
Hard admission without rehearsal compounds false evictions.

**F13c — Caveats.** (i) The seed-0 diagnostic instance hit d1 .997 while the
seed-0 arm hit .87 — same config, CUDA-nondeterministic training lands in
different basins; equilibrium selection (how much recurrence-trust survives,
where a/b calibrate) is knife-edge at 1500 steps. Treat all point values as
basin samples. (ii) The decomposition is INCOMPLETE: base-drop has only been
run with the leaky -30 masks. The critical control is base-drop with -1e9
(rehearsal alone, no gate): if it matches surp-drop-hard, the gate
contributes only telemetry; if it sits at floor, partial eviction matters.
One arm, 3 seeds:
    uv run python -m source_monitor.experiments --arms base-drop --seeds 0,1,2 --amp-n 0

**Seer translation.** Under real training pressure the architecture
rediscovered the sibling project's conclusion by gradient descent alone:
use the internal signal as a MONITOR, make robustness structural
(reduced reliance on self-reports), don't bet on reactive eviction. The
zero-shot detector (auroc ~1.0, no labels, no corruption exposure) survives
as a free abstention/verification trigger — the Inspection channel seer
wants — while the repair leg is carried by rehearsed re-derivation.

## v4 hard-admission run, seeds 0/1/2 (2026-07-17) — P9 FAIL: the 2x2 is
## fully negative, and that is theoretically suspicious

Arms surp-clean-hard / surp-drop-hard (gate_hard: gamma<0 -> -30). Detection
again ~perfect (auroc .93-.999, g_corrupt .00-.07, no false closures,
competence at ceiling). Behavior: unchanged from the blind-spot floor on all
three corruption types (ghost d1 .30-.37, misloc/phantom bsi ~.15-.17). All
four cells of {soft,hard} x {rehearsal,none} now fail behaviorally while
detection is near-perfect in every cell.

**F12 — RESOLVED: the "eviction" never evicted. -30 is not -inf.**
`diagnose.py`, surp-drop-hard config, seed 0, 300 pairs:

With the -30 bias (as run in v2/v3/v4):
  d1: GATE .337  DROP .350  CLEAN 1.000  DROP-on-CLEAN 1.000
  cos(GATE,DROP)=.984, cos(GATE,CLEAN)=.430

DROP-on-CLEAN = 1.000 proved the rehearsal DID teach perfect post-hole
re-derivation, and the same mask on a ghost trace still failed — with the
only differing token "quarantined." A deterministic model can only do that
if the quarantined content leaks. It does: the trained recurrence head
("copy the previous emission") carries attention logits comparable to or
exceeding the bias, so an additive -30 ATTENUATES rather than evicts —
invisible when the masked token is true (leak agrees), poisonous when it is
the ghost. With EVICT = -1e9:

  d1: GATE .997  DROP .997  CLEAN 1.000  DROP-on-CLEAN .997
  cos(GATE,DROP)=.9999

**The loop closes.** Detection (zero-shot surprisal, no labels) + true hard
eviction (-1e9) + rehearsed fallback (corruption-free emission dropout) =
repair at .997, on a model that never saw a corruption or a corruption label
in training.

SUPERSEDED interpretations: F9's "detection != repair / eviction is not
repair" and F11's "rehearsal does not create distrust of visible lies" were
artifacts of testing ATTENUATION while claiming EVICTION. What survives of
them: soft, logit-scale trust-discounts genuinely do not work against
specialized copy heads (their logits out-shout any bounded bias) — a real
and generalizable lesson for any "soft gating in attention" design. The
precise leak arithmetic (single-column vs pass-1 cascade) is not pinned
down and does not need to be; -1e9 removes the channel.

Definitive P9 rerun (all types, 3 seeds, both hard arms) pending below.

## v3 emission-dropout run, seeds 0/1/2 (2026-07-17) — P8 FAIL:
## rehearsing holes does not create distrust of visible lies

Arms surp-drop / base-drop (ghost_frac 0.0, dropout 0.3, hard -30 masks in
training). Means over 3 seeds.

**F11 — Repair still fails; both arms sit at the blind-spot floor.**
surp-drop: ghost d1 .30/.45/.34, misloc bsi ~.165, phantom bsi ~.163 —
base-clean numbers — while detection stays at auroc .96-.999 with
g_corrupt ≈ 0 and clean competence at CEILING (perstep 1.000: the model
demonstrably can operate across holes; dropout even improved competence).
base-drop ≡ surp-drop behaviorally, so the gate's soft eviction adds nothing
post-rehearsal. Two mechanisms, both consistent with the data: (a) CONDITION
MISMATCH — training rehearses fully-absent emissions (bias -30) but the
eval gate delivers soft suppression (logsigmoid(gamma) ≈ -3..-8), which a
strongly content-matched ghost key can punch through; (b) CATEGORY MISMATCH
— absent ≠ visibly-false: hole-rehearsal never touches the model's trust in
emissions that are present and plausible. (Also: the seed-1 partial misloc
transfer appeared AGAIN in base-drop, d1 .71 — third independent occurrence
of that more-general circuit under seed 1. Solution multiplicity is real at
this scale.) Amp instrument remains uninformative noise; excluded from
future runs (--amp-n 0).

### v4 — hard admission (closes the 2x2)

`gate_hard`: gamma < 0 -> bias -30 (binary evict), making the acted-on
condition IDENTICAL to the rehearsed one. With detection held fixed this
completes the 2x2 {rehearsal} x {hard eviction}; the two soft cells already
failed. Arms: `surp-clean-hard` (hard eviction, no rehearsal) and
`surp-drop-hard` (hard eviction + rehearsal). **P9:** surp-drop-hard reaches
d1 ≥ .9 / bsi ≤ .03 on all three types (zero corruption exposure end to
end). If surp-clean-hard also passes, rehearsal was never needed and soft
bias was the whole story; if BOTH hard cells fail, in-context eviction
cannot repair at this scale and the gate's role narrows definitively to
detection-for-abstention/verification (still a strong seer result, but the
repair leg moves to recomputation rather than masking).

## v2 surprisal-gate run, seeds 0/1/2 (2026-07-17) — P7 SPLIT verdict:
## detection transfers zero-shot; eviction does not repair

Arms surp-clean / surp-mix / base-mix, full three-type matrix (ghost /
misloc / phantom), n=1200 pairs per type. Means over 3 seeds.

**F8 — Zero-shot detection: PASS, decisively.** surp-clean — trained on ZERO
corruptions, no labels — detects all three corruption types it has never
seen: gate_auroc .970 / .977 / .966 (ghost/misloc/phantom), g_corrupt
.00-.06 (gate slams shut), g_genuine ~.88, g_clean ~.96, competence
.994-.998. Where the label-supervised gate was type-bound (held-out .68,
never closed — F4), the generative signal γ = a·logp(emitted | own
prediction) + b transfers uniformly at ~.97. The self-consistency thesis
holds for DETECTION: the model always knew the emission was wrong; the
architecture just never asked.

**F9 — Behavioral repair: FAIL — detection ≠ repair.** surp-clean behavior
is IDENTICAL to base-clean on every type (ghost d1 .31 vs .30, bsi .100 vs
.103; misloc d1 .42/bsi .170; phantom d1 .41/bsi .169): the gate closes on
the corrupted emission, yet the model still collapses at d1. Sharper:
surp-mix on held-out types — gate fully closed (g_corrupt .00, auroc
.98-.99) — behaves exactly like base-mix (misloc d1 .50 vs .55, phantom .49
vs .48). Evicting the corrupted token does not rescue the next-step
computation. Reading: post-eviction operation is itself out-of-distribution —
the model only re-derives across a masked/unreliable emission where it
PRACTICED doing so (mix training taught it for REMOVE contexts only). Repair
is a separate, also-type-bound skill that masking alone does not confer.
(Possible secondary factor: the soft bias (~logp, ≈ -5 to -8) suppresses but
is not -inf; hard-eviction-on-threshold is the contingency if v3 fails.)

**F10 — base-mix seed-1 partial transfer reproduced** (misloc d1 .71 after
.78 in the five-arm run; phantom .60): same seed, independent replication —
a reproducible alternative circuit that generalizes further, not noise.
Worth a look someday (what did s1 learn differently?), not chased now.

Amp triplet: still seed-noisy (surp-mix amp_corrupt .113/.016/.003);
remains uncitable at n=60 (F3 caveat stands).

### Interpretation for seer

The dissociation maps cleanly onto the pillars: INSPECTION is validated in
the strongest available form — a generative self-consistency signal detects
own-output corruption zero-shot, type-agnostically, at no competence cost
(exactly the transfer property the L1 token probe lacked). ADMISSION-as-
attention-eviction is NOT repair: detection must drive a different actor —
abstention / verification / recomputation (the sibling's selective-prediction
result; efh-core's gate; energy-descent correction) — not just masking. γ is
that trigger, available for free.

### v3 — emission dropout (does practiced fallback convert detection into repair?)

If repair fails because operating-across-a-hole is unpracticed, train the
fallback WITHOUT ever showing a corruption: during training, randomly
hard-mask self-emissions (key bias -30) at rate p=0.3 — self-supervised,
type-agnostic. Arms: `surp-drop` (surprise gate + dropout, ghost_frac 0.0 —
still zero corruption exposure) and `base-drop` (dropout only, no gate — the
control that asks whether hole-robust training alone dissolves the blind
spot by ending over-reliance on emissions). P8: surp-drop reaches d1 ≥ .9
and bsi ≤ .03 on ALL types zero-shot; base-drop alone does not (it lacks
eviction at test time... unless emission-redundancy makes corruption
harmless — either outcome pins down the mechanism).

## Five-arm run, seeds 0/1/2 (2026-07-17) — verdicts on P1–P5

Config: defaults (d256/L6, Muon, 1500 steps, ghost_frac 0.3, n=1200 eval
pairs per corruption type, amp n=60). All numbers seed0/seed1/seed2.

**F1 — P1 replication: PASS, tight.** base-clean ghost: follow
.200/.201/.196, bsi .105/.103/.102, d1 .29/.29/.31 — the sibling's anchors
(0.203 / 0.111 / 0.28) reproduced across seeds in a different codebase.
Instruments trusted. New anchor: mislocation is intrinsically HARDER than the
ghost (base-clean bsi .165 vs .103, d1 ~0.43 vs 0.30) — a plausible lie
(wrong container, object present) is costlier than an impossible one
(container after REMOVE), presumably because the REMOVE contradiction is
readable from the immediately preceding op.

**F2 — P2 behavioral margin: NULL. Data exposure alone solves the
trained-on type.** Every mix-trained arm crushes the ghost: base-mix d1
.97/.96/.98 (bsi ≤ .007), prov-mix ~.97, gate-task ~.98, gate-sup ~.99.
The SPEC §7 risk materialized exactly: at this scale, seeing corrupted
emissions in training teaches re-derivation without any mechanism. No
behavioral advantage for the gate on the corruption type it was trained on.

**F3 — P3 mechanism: the supervised gate works exactly as designed,
in-type.** gate-sup ghost detection: gate_auroc 1.000/.999/1.000,
g_corrupt .01/.06/.00, g_genuine 1.00, g_clean 1.00, competence
.990-.999 — a perfect, calibrated, inspectable trust dial with zero
competence tax and zero false closures. The spectral half of P3 is
INCONCLUSIVE: the JVP amp numbers swing by seed (control alone ranges
.042-.183; base-clean amp_genuine .29 on s1 vs .04 on s2) — n=60 at this
scale cannot support the selective-contraction claim. Do not cite the amp
triplet either way (sibling's "amp estimator ceiling" caveat, confirmed).

**F4 — P4 transfer: FAIL, everywhere — the central result.** Held-out
mislocation, gate-sup: gate_auroc .646/.719/.674 (predicted ≥ .8),
g_corrupt 1.00/.92/1.00 — the gate carries weak *ranking* signal but never
actually closes on a corruption type it wasn't trained on. Behavior matches:
misloc d1 ~.46-.55 and bsi ~.14-.15 for gate-sup, indistinguishable from
base-clean (.43 / .165). And SYMMETRICALLY: base-mix's data-robustness also
fails to transfer (misloc bsi .169/.058/.165 vs its ghost bsi ≤ .007).
Ghost-trained robustness — whether learned from data or supervised into a
gate — is CORRUPTION-TYPE-SPECIFIC. (base-mix s1 is a lone partial-transfer
outlier, d1 .78 / bsi .058: seed-level solution multiplicity; noted, not
chased — the other five mix-arm seeds cluster at no-transfer.)

**F5 — P5 emergence: NULL.** gate-task auroc .306/.339/.559 (chance-ish),
g ≡ 1.00 everywhere: task loss found the re-derivation solution without ever
touching the gate. No emergent source monitoring at this scale. Behavioral
robustness in that arm comes from the mix data, not the mechanism.

### Interpretation

This is the sibling's L1 probe-transfer failure REPRODUCED AT THE MECHANISM
LEVEL in a fully controlled setting: discriminatively-supervised admission
(BCE on "was this emission corrupted?") learns the trained corruption's
signature, not the general property "inconsistent with state re-derived from
external evidence." The seer-relevant conclusion is sharp: any
label-supervised trust channel should be expected to be type-bound; a
transferable admission signal must come from a GENERATIVE/self-consistency
quantity — score the emission against the model's own re-derived belief —
not from corruption labels. That is precisely the energy-channel bet, and it
is now testable in this harness (v2 below).

### v2 — the surprisal gate (SPEC §9 addendum)

`gate="surprise"`: γ_i = a·logp(emitted token_i | model's own final-layer
prediction at the marker) + b, learnable scalars (a,b), no labels, no
corruption exposure needed. Sequence-causal (the marker state precedes the
emission), two-pass in parallel training, one streaming pass at inference.
By construction corruption-type-agnostic: ANY false emission is low-probability
under correct state tracking. New arms `surp-clean` (ghost_frac 0.0 — never
sees a single corruption) and `surp-mix`. Prediction: surp-clean detects and
discounts ALL THREE corruption types (ghost, misloc, and the new held-out
phantom-removal) at gate_auroc ≥ .9 with d1 ≥ .9 — beating gate-sup's
transfer while using strictly less supervision. If it does, the
generative-signal thesis stands; if it also fails, admission-by-internal-
signal is in trouble at this scale generally.

## Scaffolded (2026-07-17)

Project created: model (provenance + depth-causal admission gate), ghost-mix
training, both corruption injectors, vendored instruments, five-arm
orchestrator, guardrail tests. Milestone 1 (pytest) passed same day; five-arm
run above.
