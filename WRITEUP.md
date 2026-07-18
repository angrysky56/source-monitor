# Distrust by Default: Self-Monitoring Without Supervision

**source-monitor — findings report.** 2026-07-17. One day, one RTX 3060,
4.9M-parameter models, ~40 training runs, 9 seeds at the end.
Chronological lab log with every dead end: FINDINGS.md. Raw numbers:
results/results.jsonl.

## Abstract

A small autoregressive model that emits its own state reports and reads
them back exhibits the self-correction blind spot: one false self-emission
collapses next-step accuracy from ~1.0 to 0.30, because the model treats
its own output as fresh external evidence. We ask what architectural
ingredients remove this, and find: (1) label-supervised trust learns the
corruption's signature and does not transfer to unseen corruption types;
(2) a label-free generative signal — the surprisal of the model's own
emission under its own prediction — detects every corruption type
zero-shot, with no corruption exposure in training, at no competence cost;
(3) evicting a distrusted token from attention requires a -inf-grade
intervention, because trained copy heads carry logits that shout through
any bounded bias; (4) behavioral repair is carried primarily by REHEARSAL —
training with random hard-masked self-emissions builds a re-derivation
circuit and structural distrust-by-default — with detection-triggered
eviction adding a modest increment that reaches significance on the hardest
held-out corruption type (bsi +.037, p=.023, n=9 paired seeds); and (5)
when gates are behaviorally idle, their calibration is not sign-anchored
and must be constrained structurally. Given the choice, gradient descent
itself converges on "reduce reliance on self-reports and keep the detector
as telemetry" rather than "detect and evict."

## 1. Setup

**Task.** Entity tracking (PUT/MOVE/REMOVE over objects and containers)
with dense per-step state emission: after every operation the model emits
the tracked object's current location, and that emission is fed back into
the context as an input token. The final emission is the answer. Vendored
from `sps-blindspot`, which validated the harness and the baseline numbers.

**Corruptions (teacher-forced, eval-time).** Ghost: after the tracked
object is REMOVEd, its emission is falsified to a container. Mislocation:
a present object's emission points to the wrong container. Phantom
removal: a present object's emission says NOWHERE. Only the ghost type is
ever seen in training (and only in designated arms); mislocation and
phantom are held out always — the transfer probes.

**Instruments.** Behavioral: ghost_follow, post-corruption accuracy,
blindspot index (bsi = clean minus corrupted final accuracy), and the
recovery-by-distance curve whose first point (d1) is the signature
statistic. Mechanistic: per-position gate values and detection AUROC;
a JVP Jacobian estimator (which proved too seed-noisy to use at n=60 and
is excluded from conclusions). All arms share backbone (d256/L6, ~4.9M),
optimizer (Muon+AdamW split), data, and steps (1500); arms differ only in
the mechanism under test. Nothing was believed until it survived 3 seeds;
the headline comparison ran 9.

## 2. Results

**R1 — The blind spot is real and graded by plausibility.** Clean-trained
baseline: d1 = 0.30, bsi = .10 on ghosts; the model's next prediction is
dominated by its own false emission, then recovers by re-deriving from the
operations at d2. Plausible lies cost more than impossible ones
(mislocation bsi .165 vs ghost .103): the REMOVE contradiction is readable
from the immediately preceding operation; a wrong-but-plausible container
requires longer-range reconciliation.

**R2 — Supervised trust does not transfer.** A learned admission gate
trained with corruption labels (BCE on "was this emission falsified?")
becomes a perfect calibrated in-type detector (AUROC 1.000, zero false
closures, zero competence tax) — and never closes on corruption types it
was not trained on (held-out AUROC ~.68, g_corrupt ~1.0). Symmetrically,
robustness learned from corrupted-data exposure alone is equally
type-bound. Discriminative supervision learns the trained corruption's
signature, not the general property "inconsistent with state re-derived
from external evidence." This reproduces, at the mechanism level, the
domain-transfer failure that killed the token-space confidence probe in
the sibling project (L1).

**R3 — Generative self-consistency transfers, zero-shot.** Replace the
supervised gate with γ = softplus(a)·logp(emitted token | the model's own
prediction at the emission point) + b — two calibration scalars, no
labels. Trained on entirely clean data (zero corruptions ever seen), this
detects ALL THREE corruption types at AUROC .93-1.00 with corrupted
emissions driving the gate toward closed and genuine ones leaving it open.
The model always contained the information that its emission was wrong;
the architecture had simply never asked. Detection is the cheap,
transferable part of self-monitoring.

**R4 — Eviction must be -inf-grade.** Three successive negative results
("eviction doesn't repair", "rehearsal doesn't transfer", "even hard
eviction fails") dissolved under one diagnostic: an additive attention
bias of -30 does not evict a token whose trained copy-head logits are
comparable to the bias — it attenuates, and the leak is invisible when the
masked token is true and poisonous when it is false. The four-way
diagnostic (gate vs forced-mask vs clean vs mask-on-clean) localized this
precisely: with -1e9, gate and forced-mask agree to cosine 0.9999 and d1
goes 0.35 → 0.997. Design rule: soft, bounded "trust discounts" inside
attention lose to specialized heads; gate to effective -inf or put the
gate somewhere else.

**R5 — Repair is carried by rehearsal; the gate adds a real but modest
increment; and optimization prefers monitoring to eviction.** Emission
dropout — randomly hard-masking 30% of self-emissions during training,
still zero corruptions — forces a first-class re-derivation circuit.
Rehearsal alone (no gate, no provenance) cuts bsi from .10-.17 to
.02-.07 across all types. Adding detection-triggered hard eviction yields,
over 9 paired seeds: nothing on ghosts (both at floor), +.029 bsi on
mislocation (p=.121, 6/9 seeds), and +.037 bsi on phantom (p=.023, 8/9
seeds) — roughly halving residual damage on the held-out types, significant
on one of the two pre-registered discriminators. Strikingly, in the
combined arm the learned calibration keeps gates largely OPEN on corrupted
positions (ranking preserved at AUROC ~1.0): given a robust fallback,
gradient descent stops paying eviction's false-positive cost and keeps the
detector as pure telemetry. Reactive eviction without rehearsal, by
contrast, cascades: contaminated beliefs make the next TRUE emission look
surprising, which gets evicted in turn (a transient d2 dip that self-heals
as beliefs re-derive).

**R5b — Idle gates lose their sign.** In one of nine seeds the calibration
inverted (detection AUROC .003 — perfect reversed ranking) with no
behavioral cost, because open-gate equilibria give (a, b) almost no
gradient. Fixed structurally: a = softplus(a_raw); surprisal can only ever
close a gate. Telemetry from before the fix is unsigned ranking.

## 3. The graveyard (what died, and what killed it)

Five headline conclusions were drafted and destroyed during this project:
the "Detectability Paradox" (sibling project; died by faithful lens +
multi-seed), "SPS shrinks the blind spot" (died by H1/H2 instruments),
"eviction does not repair" and "rehearsal does not transfer" (both died by
the -30 diagnostic), and "the 2x2 is fully negative" (same). The surviving
conclusions are the ones above. The method lesson is not decoration; it is
the main reason the final claims are trustworthy: instrument before
conclusion, control before mechanism story, seeds before belief. Basin
variance at this scale (1500 steps, 4.9M params, CUDA nondeterminism) is
the dominant noise source and repeatedly produced reproducible-looking
single-seed effects, including one seed that reliably learns a
more-general re-derivation circuit in three different arms.

## 4. Design directives

For any system that re-reads its own outputs: (1) keep provenance — the
runtime always knows which tokens it emitted; do not discard the bit.
(2) Get detection from a generative self-consistency signal (surprisal of
own output under own predictive state), sign-anchored, calibrated with an
affine — never from corruption labels. (3) Make robustness structural:
train under self-report blackouts so operating-without-trusting is
in-distribution. (4) Put thresholds on the ACTION side — abstain, verify,
recompute — where a false positive costs an extra check, not a belief;
if in-context eviction is used at all, it must be true removal, not a
bias. (5) Expect optimization to prefer default-distrust over reactive
gating whenever both are available; design for the detector as telemetry
feeding an outer loop.

## 5. Limitations

Single synthetic task; 4.9M parameters; 1500 steps; teacher-forced
corruptions rather than free-running self-errors; the mislocation
increment is underpowered (n=9); the JVP spectral instrument never rose
above noise; no scale transfer evidence. Every claim is a claim about this
harness until ported (see SEER-INTEGRATION.md for the port).

## 6. Code map

model.py (provenance embedding, learned/supervised gate, surprisal gate,
hard admission, emission-drop bias) · task.py (task + three injectors) ·
train.py (ghost-mix + dropout training) · blindspot.py (behavioral
protocol + gate diagnostics) · diagnose.py (the four-way F12 discriminator)
· amplification.py (JVP; retired) · experiments.py (arms × seeds → JSONL)
· f14_power.py (paired exact sign-flip analysis).
