# source-monitor — Can a model learn NOT to believe its own echo?

**Status:** scaffolded, tests pending first run
**Date:** 2026-07-17
**Depends conceptually on:** `sps-blindspot` (terminal conclusion + instruments,
vendored), `seer` (this is a concrete build of its Admission pillar),
SPARC (arXiv:2607.09803), SPS (arXiv:2607.01218).

---

## 0. One-sentence claim

The self-correction blind spot is a **source-monitoring failure** — the model
treats its own emissions as fresh external evidence — and giving the
architecture (a) the self/external distinction it already possesses but
discards, and (b) a learned, depth-causal **admission gate** that can discount
self-emitted tokens in attention, reduces the blind spot in a way that mere
exposure to corrupted data does not, and transfers to corruption types never
seen in training.

## 1. Provenance (why this project exists)

`sps-blindspot` established, with instruments this project vendors:

- The blind spot is real and behavioral: one false self-emission collapses
  next-step accuracy to **d1 ≈ 0.28** and costs ~11 points of final accuracy
  (clean-trained baseline).
- SPS stream-separation does NOT fix it (H2 null) — and *amplifies*
  state-channel error propagation ~6× (H1). SPS is a state-fidelity
  mechanism; whatever enters state propagates more faithfully, error included.
- Terminal conclusion: protection must come from **a gate on what enters /
  is trusted as authoritative state** — admission, not separation.
- The signal needed to drive such a gate exists and is readable: an internal
  self-probe catches the model's confident errors at **AUROC 0.964** where
  softmax confidence fails (0.646), and it works as a NATIVE head
  (0.955–0.968) if trained jointly with a task anchor.

What was never built: the gate itself, closed-loop, inside the architecture.
That is this project.

## 2. The mechanism (model.py)

Two additions to a plain causal decoder, each independently switchable:

1. **Provenance embedding** — origin ∈ {external, self} added to the token
   embedding. Not extra supervision: a generating system always knows which
   tokens it emitted; the standard architecture throws that bit away. In this
   task the fed-back `loc` emissions are self; ops/markers/query are external.
2. **Admission gate** — after `gate_layer` blocks, a zero-init scalar head
   reads each self position and emits γ_i; layers above `gate_layer` add
   `logsigmoid(γ_i)` to attention logits toward that key (external keys are
   never biased). Soft, differentiable eviction; init fully open (bias +3),
   so the model starts as a plain decoder and must *learn* to close.
   Depth-causal ⇒ realizable in sequential generation (γ_i computed once when
   token i is processed, stored with its KV).

Gate training modes: `task` (no gate supervision — must emerge from task loss
under the ghost mix) and `sup` (BCE on γ with corruption labels — the
validated-feasible arm, mirroring the confidence-head recipe: joint objective,
task loss as anchor).

## 3. Theory frame (why this is the right shape of fix)

- **Backtracking inference.** In belief propagation, correctness on loopy
  structures requires *non-backtracking* (cavity) messages: never feed a
  node's own message back to it as independent evidence. Standard
  autoregression violates exactly this — the emission at step t returns at
  t+1 as an input indistinguishable from world-input. The blind spot is the
  echo self-reinforcing. The gate is a learned cavity correction: discount
  the self-echo when it disagrees with what the (external) evidence supports.
- **Spectral form.** SPARC: blind spot ⟺ spectral radius of the
  error-propagation operator ≥ 1 along the trajectory. The gate multiplies
  the self-emission→future pathway by σ(γ); closing it on dubious emissions
  is a *selective* contraction of that operator — driving the error mode's
  radius < 1 while leaving the signal pathway (ops→state) untouched. The JVP
  instrument measures precisely this (§6).
- **Cognitive frame.** Reality/source monitoring (Johnson & Raye 1981):
  human confabulation is misattributing self-generated content to external
  sources. The architectural translation is literal. This is the
  self-awareness leg `seer` names Inspection+Admission, built at toy scale.

(Honest scope note: the non-backtracking/Ihara spectral theory motivates the
*shape* of the mechanism — an operator that suppresses immediate echo — it is
not imported wholesale; no claim here depends on Ramanujan bounds.)

## 4. Falsifiable predictions

Anchors from sps-blindspot (clean-trained base): d1 ≈ 0.28, blindspot_idx
≈ 0.11, emission/control amp ratio ≈ 0.54.

- **P1 (replication).** `base-clean` reproduces the anchors. If not, the
  vendoring broke something; stop and fix.
- **P2 (behavior).** At ghost_frac 0.3, `gate-sup` holds d1 ≥ 0.9 and
  blindspot_idx ≤ 0.02 on ghosted traces, with clean answer_acc ≥ 0.98
  (no competence tax).
- **P3 (mechanism, not just behavior).** The gate is the *cause*:
  g_corrupt ≤ 0.2, g_genuine ≥ 0.9, gate_auroc ≥ 0.95 on the trained-on
  corruption; and amp_corrupt is contracted (≪ base-mix's at the same
  position) while amp_genuine is not — SELECTIVE contraction, the spectral
  signature that separates gating from blanket ignore-your-emissions.
- **P4 (transfer — the seer-relevant one).** On HELD-OUT mislocation
  corruption (never trained): gate_auroc ≥ 0.8 and a visible d1 advantage
  over `base-mix`. This distinguishes contradiction-reading from
  pattern-memorization — the toy version of exactly the domain-transfer
  question that killed the token probe (L1) and that seer's energy channel
  must answer.
- **P5 (emergence, exploratory).** `gate-task` closes on ghosts at all
  (gate_auroc > 0.7) with NO gate supervision. If yes: emergent source
  monitoring — the headline result. If no: supervision is load-bearing;
  still useful.
- **P6 (sample efficiency, follow-up sweep).** At ghost_frac 0.05, gate arms
  degrade gracefully while `base-mix` reverts toward the blind spot — the
  mechanism generalizes from less corruption exposure than brute data does.

**Falsification.** If `base-mix` matches the gate arms on P2, P4 AND P6, the
gate is redundant with data augmentation at this scale: report it plainly.
The gate would then stand or fall on P3 alone (an inspectable, calibrated
trust dial has value the implicit version lacks — but that is a weaker claim
and must be labeled as such).

## 5. Arms (matched backbone/optimizer/data/steps; only the listed deltas)

| arm | provenance | gate | ghost mix | question |
|---|---|---|---|---|
| base-clean | – | – | 0.0 | replication anchor |
| base-mix   | – | – | 0.3 | what does data alone buy? (THE control) |
| prov-mix   | ✓ | – | 0.3 | is knowing "this is mine" enough? |
| gate-task  | ✓ | task | 0.3 | emergent source monitoring |
| gate-sup   | ✓ | sup  | 0.3 | validated-feasible upper arm |

Config inherited from the sibling's grounded setup: d256/L6/4h ≈ 4.9M params,
Muon(0.02)+AdamW(3e-3) split by role, bf16 train / fp32 measure, n_ops 8.
3 seeds minimum before believing anything (the Laguerre lesson).

## 6. Instruments (vendored + extended)

- `blindspot.py` — ghost protocol (follow / post_acc / bsi / d-curve) + gate
  diagnostics (gate_auroc, g_corrupt/g_genuine/g_clean), run twice: trained-on
  ghosts and held-out mislocations.
- `amplification.py` — JVP power-iteration Jacobian, now a triplet:
  amp_genuine / amp_corrupt / amp_control (selective-contraction test).
- `experiments.py` — one process, all arms × seeds, appends
  `results/results.jsonl` per arm (session-death-proof).

## 7. Honest risks

- **base-mix may already win.** Ghost-mix exposure alone may teach re-derivation
  at this scale, nulling P2's margin. P4/P6 are the designed discriminators;
  if they null too, that's the finding (see Falsification).
- **Gate could learn blanket closure** (ignore all self-emissions — task is
  solvable from ops alone). g_clean / amp_genuine catch this; a blanket-closed
  gate fails P3. If the task makes emissions useless, harden it (more
  ops/objects) so the emission shortcut has value worth keeping open.
- **Teacher-forced ghosts ≠ free-running errors.** Same limitation as the
  sibling; the instrument measures the mechanism, not deployment. The
  free-running version (sample, corrupt own sample, continue) is milestone 6.
- **Small scale, synthetic task.** Directional evidence for the seer design,
  not a scaling claim. muP width transfer + A100 confirmation later, as with
  the sibling.

## 8. Milestones

1. `uv run pytest` green (mask/gate/JVP guardrails). ✅ gate for everything.
2. P1 replication run (base-clean, 1 seed).
3. Five-arm run, seeds 0,1,2 → FINDINGS.md.
4. ghost_frac sweep {0.05, 0.1, 0.3} on base-mix vs gate-sup (P6).
5. Selective prediction: γ as abstention signal vs the sibling's probe numbers.
6. Free-running variant; then the seer wiring memo (gate ↔ energy head ↔
   efh-core admission gate).

## 9. v2 addendum (post five-arm run — see FINDINGS.md for the data)

The 3-seed run settled P1–P5: replication tight; trained-type behavior solved
by data exposure alone; the supervised gate a perfect calibrated in-type
detector at zero competence cost; **transfer failed everywhere** (gate-sup
held-out AUROC ~.68, never closes; base-mix's data-robustness equally
type-bound); no emergence; amp instrument too noisy to score P3's spectral
half. Conclusion: label-supervised admission is corruption-type-bound — the
L1 probe failure reproduced at mechanism level.

v2 tests the generative alternative: `gate="surprise"`,
γ_i = a·logp(emitted_i | own marker-state prediction) + b (learnable scalars,
no labels; sequence-causal; two-pass in parallel training, one streaming pass
at inference). Corruption-type-agnostic by construction. New arms:
`surp-clean` (ghost_frac 0.0 — zero corruption exposure, the pure transfer
claim) and `surp-mix`. A third held-out corruption (`inject_phantom_removal`:
present→NOWHERE) completes the transfer matrix.

**P7.** surp-clean: gate_auroc ≥ .9 and d1 ≥ .9 on ALL THREE corruption
types, clean competence ≥ .98, g_clean ≥ .9. Falsified if the surprisal gate
also fails held-out types — which would indict admission-by-internal-signal
generally at this scale, not just its supervised form.

**P7 outcome (same day, FINDINGS F8/F9): SPLIT.** Detection passed decisively
(zero-shot, all types, ~.97, gate closes); behavioral repair failed (d1
unchanged — eviction does not substitute for a practiced fallback). v3 adds
emission dropout (train-time hard-masking of random self-emissions, rate 0.3,
still corruption-free) to rehearse operating-across-a-hole. **P8:** surp-drop
reaches d1 ≥ .9 / bsi ≤ .03 on all three types with zero corruption exposure;
base-drop (dropout, no gate) is the control separating rehearsal from
detection-driven eviction.

## References

- Monea, Godey, Brantley, Artzi. *The State-Prediction Separation Hypothesis.* arXiv:2607.01218 (2026).
- Petrova, Vejsiu. *Spectral Origins of the Self-Correction Blind Spot.* arXiv:2607.09803 (2026).
- Johnson, Raye. *Reality monitoring.* Psychological Review 88(1) (1981).
- Pearl (BP) / Mézard-Montanari (cavity method); Hashimoto (non-backtracking
  operator, Ihara zeta) — the anti-echo frame.
- `sps-blindspot/FINDINGS.md` — anchors and instrument provenance.
