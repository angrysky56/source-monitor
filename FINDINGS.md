# source-monitor — findings log

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
