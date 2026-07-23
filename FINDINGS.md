# source-monitor — findings log

## F27 — Build A: the router shipped & validated (2026-07-23; Qwen3-1.7B,
## entity_prose + factual_qa, n=40). Content-only classification enforces clean
## separation without oracle cheats.

Architecture context: Combines Leg 1 (retrospective surprisal; context-derivable errors)
and Leg 2 (sampled consistency; factual recall/confabulation).

Built: `loop/router.py` (content-only `is_context_derivable`, `route`, independent binary flag evaluation),
`loop/f27_router.py` (runner + 3 control tracks + per-span audit logger), `tests/test_router.py` (CPU tests).

Key Results & Constraints Verification:
- **Content-Only Classifier (`is_context_derivable`):** Uses word-boundary matching on user/assistant dialogue turns (excluding system instructions and negation candidates). Zero metadata (`trace.meta["grounded"]`) inspection.
- **Track 1 (All-Derivable Control - `entity_prose`):** 100.0% classified content-derivable -> 100.0% routed to Leg 1 (surprisal). Identity Control Match (Router == Leg 1): **100.0%** (PASS).
- **Track 2 (All-Factual Control - `factual_qa`):** 91.7% classified content-underivable -> 91.7% routed to Leg 2 (consistency). Identity Control Match (Router == Leg 2): **91.7%** (PASS; 1 boundary case correctly routed to Leg 1 because context contained "paris").
- **Track 3 (Provisional Mixed Eval - Planted Lies):** Evaluated mixed corpus (clean vs planted wrong-value claims).
  - Routed Monitor AUROC: **0.460** (Provisional).
  - Leg 1 AUROC: **0.714** | Leg 2 AUROC: **0.429**.
  - Catch rates: Routed 11.1% (False flag 19.0%) vs Leg 1 100.0% (False flag 57.1%) vs Leg 2 0.0% (False flag 14.3%).
  - Confirms user review note: factual positive-class design on recall tasks remains provisional and needs reviewer task co-design (Build B).

---

## F26 — the factual leg (consistency detector): teacher-forced preference reads
## the PRIOR, not knowledge (dead end); SAMPLED self-consistency DELIVERS the leg
## (AUROC .82 vs .50). Both monitor legs now exist. (2026-07-21; Qwen3-1.7B,
## factual_qa, n=60.)

Architecture context. F25 located seer's real gap: errors NOT contradicted by
context — factual recall, where F19 already put surprisal at ~chance. Target: the
SECOND leg of a routed monitor — context-derivable claims -> surprisal (F21e/F25);
factual claims -> consistency (this). Mechanism aimed at: SelfCheckGPT — a known
fact is answered consistently, a confabulation varies.

Built: `loop/consistency.py` (query-paraphrase family + preferred-value +
answer-stability), `loop/f26_consistency.py` (viability: known vs unanswerable on
factual_qa, grounded=False), `tests/test_consistency.py` (6 CPU tests). Reuses
`base.make_variant` / `raw_claim_score` and F22's `auroc`.

**F26a — two scoring biases the smoke caught (the scaffold earning its keep).**
(1) Whole-claim `mean_neglogp` is length-biased toward the long negation "I have no
reliable record", so the model spuriously "prefers" abstention for EVERY question
(known modal_correct 0.000). Fixed -> value-slot scoring. (2) Even slot-scored, the
abstention still dominates because the SYSTEM PROMPT primes it (uniformly low
surprisal). Fixed -> exclude negation, compare VALUE candidates only. Both are the
Phase-0b negation-prior bias resurfacing in a new place.

**F26b — THE finding: teacher-forced preference is stable BY CONSTRUCTION, so it
cannot carry a consistency signal.** With the biases removed, the model's preferred
VALUE is stability = 1.000 across paraphrases for BOTH known and unanswerable
questions (AUROC 0.500 for every consistency signal). This is structural, not a
tuning miss: argmin over teacher-forced candidate surprisal reads the PRIOR token
ranking (Paris > Lyon by frequency), which shallow query paraphrases do not move —
independent of whether the model "knows". Consistency-of-teacher-forced-preference
measures prior stability, not knowledge.

**F26c — the viable instrument is SAMPLING (real SelfCheckGPT), not teacher
forcing.** What distinguishes known from confabulated is SAMPLING VARIANCE:
generate k answers at temperature and measure agreement — a known fact is generated
consistently, an unknown one varies. Teacher-forced argmax collapses that variance
by construction. So the factual leg needs a generation loop + answer
normalization/matching — a heavier, different mechanism than the teacher-forced
rest of the monitor. That is a real architectural fork: legs 1 (surprisal) and 2
(consistency) do not share an instrument.

**Consequence / next build.** Keep the query-paraphrase family + eval harness
(reusable); REPLACE `preferred_candidate` (teacher-forced) with `sample_answers`
(temperature generation) + string-normalized agreement. Pre-register:
agreement(known) > agreement(unanswerable) at AUROC >= .65 while min-surprisal is
~chance. Then wire the router (context-derivable -> leg 1; factual -> leg 2).

**F26d — the sampling instrument DELIVERS (built + run: `sampled_consistency`,
`f26_sample.py`, n=60, k=6, temp 0.8).** Drawing k independent generations and
measuring whether the answer holds its shape separates unanswerable from known at
AUROC **.818** (hedge_rate) / **.761** (distinct_ratio) / .760 (disagreement) —
against teacher-forcing's .500. known: distinct .175 (6 samples ≈ 1 answer), hedge
.000, correct 1.000; unanswerable: distinct .545, hedge .545. Both gates PASS
(H-samp-1 .818; H-samp-2 known-correct 1.000). Answers audited, not a black box:
"capital of Canada?" -> all 6 agree and are correct; "grains of sand on Brighton?"
-> all 6 stably abstain. The discrete-symmetry instrument (independent draws) works
where the continuous one (smooth paraphrase) could not — the Noether/rigidity
framing, cashed out.

Honest caveats: (1) the STRONGEST signal is hedge_rate — factual_qa's unanswerable
set is OBVIOUSLY unanswerable (Caesar's breakfast), so the win is partly calibrated
REFUSAL (content), not sampling variance. (2) distinct_ratio (.761) is the GENERAL
signal — pure sampling variance, works without hedging — and it passes; that is the
true consistency detector. (3) The dangerous case — confident CONFABULATION on
plausible-but-unknown questions — is under-represented here; distinct_ratio is its
signal but this task does not stress it. (4) n=60, 1 seed; firm with multi-seed and
a harder confabulation task.

**Architecture status: BOTH legs now exist.** Leg 1 (surprisal, context-derivable)
is near-perfect (F21e/F25); leg 2 (sampled consistency, factual) is viable (here).
The two legs use DIFFERENT instruments (teacher-forced scoring vs temperature
sampling) — a real fork, not a nuisance. Missing piece = the ROUTER: per span,
decide "is the answer present in context?" and dispatch to leg 1 or leg 2. That,
plus a harder confabulation task, is the path to a functional two-leg monitor.

Files: `loop/consistency.py` (teacher-forced + sampled), `loop/f26_consistency.py`
(the negative), `loop/f26_sample.py` (the positive), `tests/test_consistency.py`
(10 CPU tests). Raw: `results/llm_f26_sample_results.jsonl`.

## F25 — Hard-task setup for the dispersion axis: single-pass CEILINGS on
## in-context-verifiable tasks (short AND long ξ), so the murmuration axis has no
## headroom to exploit. The tasks with headroom are factual — where surprisal is
## the wrong tool (F19d). (2026-07-21; infra + headroom probe, Qwen3-1.7B.)

Built the hard task F24d asked for:
- `arithmetic.generate` gains `corrupt_mid`: plants a SUBTLE off-by-(1..3) wrong
  running total in a non-first emission; the final claim stays correct; mid
  emissions always carry location_text so planted/clean score identically.
  `corrupt_mid=False` is byte-unchanged (F19 arithmetic results safe).
- `paraphrase.py` generalized: per-task `rerender_fn`; `rerender_arithmetic`
  rewords the running-total FRAMES (RT_POOL), holding the numbers. entity_prose
  behaviour identical (`rerender` alias).
- `f24_murmuration --task {entity_prose,arithmetic}` and `--n-ops` (chain length /
  difficulty). 13 CPU tests (8 entity + 5 arithmetic) pass; identity control exact.

**F25a — the hard task is not hard for single-pass (headroom probe: arithmetic
n_ops=9, off-by-small corruption, n=30, seed 42).** single-pass AUROC .992, catch
1.000. Even a long-ξ running total with a SUBTLE ±(1..3) mid error is caught
essentially perfectly — Qwen3-1.7B reads the in-context inconsistency directly.
Same ceiling as entity_prose (F24), for the same reason. My "long-ξ gives
headroom" hypothesis (F24d) is empirically FALSE for this model/task.

**F25b — the reframe (the real finding): surprisal ceilings whenever the lie is an
in-context INCONSISTENCY — which is every synthetic task in this repo.** A
teacher-forced score reads the contradiction between the claim and its context
tokens directly, so there is almost nothing left for ensembling (F23) or
dispersion (F24) to add — no headroom by construction. F23's +.094 and F24's
H-mur-1 were real but SMALL precisely because the single-pass baseline is already
near-ceiling.

**F25c — where headroom actually lives: errors NOT contradicted by context.** A
factual hallucination (no in-context ground truth) or a self-consistent reasoning
slip is exactly where single-pass surprisal is weak — and that is the regime F19d
already routed to RETRIEVAL, not surprisal. So the dispersion axis can only add
value where surprisal is the wrong detector anyway. Blunt version: on
context-verifiable tasks the murmuration idea is a solution without a problem;
single-pass surprisal already wins.

**Consequence for seer.** The monitor's surprisal leg is near-perfect for
context-contradicted errors and needs NO ensemble/dispersion booster there. F23's
--confound and any further murmuration sweeps on entity/arithmetic would be
polishing a ceiling. Effort belongs on the OTHER leg — factual claims -> retrieval
(F19d) — and, only if the dispersion axis is worth testing at all, on a task with
genuine model UNCERTAINTY: uncomputable arithmetic (large products the model
cannot verify), multi-object working-memory overload, or free-form reasoning.

Files: `ood/arithmetic.py` (corrupt_mid), `loop/paraphrase.py` (pluggable),
`loop/f24_murmuration.py` (--task/--n-ops), `tests/test_paraphrase.py` (13 tests).
No results jsonl kept — F25a is a probe, its numbers live here.

## F24 — "murmuration of a claim": paraphrase-family scoring. Dispersion carries
## real signal (H-mur-1 PASS) but is largely redundant with the mean at this
## ceiling (H-mur-2 FAIL, as pre-registered) (2026-07-21; Qwen3-1.7B, 30x2, k=6).

    detector    auroc    catch   Δ vs single   cleanFP
    single      0.9950   0.900   +0.000        0.017
    fam_mean    0.9977   0.950   +0.050        0.033
    fam_std     0.7076   0.033   -0.867        0.050
    fam_c0.5    0.9973   0.967   +0.067        0.033
    fam_c1.0    0.9964   0.967   +0.067        0.033

    GATE: control PASS · H-mur-1 PASS (AUROC fam_std .708 >= .60) ·
          H-mur-2 FAIL (best +.067 over single OK, but +.017 over fam_mean < .02)

Idea (the F22/F23 dual-representation arc, made testable): a hallucination is a
fragile fixed point — locally plausible but not robustly supported — so its
surprisal should be UNSTABLE when you reword its support, while a grounded claim
stays stable. Score each span across a faithful paraphrase family (reword the
assistant's acks, hold facts + the scored span + user instructions), and read the
family MEAN and STD. Where F23 perturbed weights, F24 perturbs phrasing.

**F24a — the idea has real substance (H-mur-1 PASS).** Dispersion alone separates
lies from clean spans at AUROC .708 — well above chance. Lies genuinely are more
unstable when their support is reworded, exactly the dual-representation
prediction. The phenomenon is real and measurable. (fam_std's catch is .033
because as a STANDALONE trigger its argmax/floor don't isolate the lie; AUROC is
the right measure of "carries signal," and .708 is a true positive.)

**F24b — looking at the family beats looking at the instance.** fam_mean
(paraphrase-averaged surprisal) lifts catch .900 -> .950 over single-pass, and
mean+σ reaches .967 (+.067). Same shape as F23: the aggregate over a family is a
better detector than the single instance. The murmuration framing produces a real
gain.

**F24c — THE HONEST FINDING (pre-registered): the gain is the MEAN, not the
dispersion.** Almost all of it (+.050 of +.067) is paraphrase-AVERAGING; the
instability term adds only +.017 over the mean, under the +.02 bar. Dispersion is
largely redundant with surprisal — a surprising span is also an unstable one. I
flagged this exact risk in F22 before running. The strong claim — that
paraphrase-instability is a DISTINCT detector axis beyond surprisal — is NOT
established.

**F24d — and it is a CEILING / POWER problem, not a clean refutation (F21b,
again).** Single-pass is already AUROC .995 / catch .900 here — ~.05 of total
headroom. The +.02-over-mean bar lives inside sampling noise: at n=60 one trace =
.017 catch, so fam_mean (.950) vs mean+σ (.967) differ by ONE trace. The marginal
value of dispersion is statistically indistinguishable from zero on this task —
not because it is absent, but because the task is too easy to resolve it. The
clean test of the dispersion axis needs a HARDER task with real single-pass
headroom (long-ξ arithmetic; factual-retrieval), at F23 scale (60x3).

**F24e — structural blind spot of the dispersion signal.** A FIRST-position lie
has no prior support to reword, so its family std is 0 by construction (debug:
lead-off lie std .00 while its mean stayed 25.4). Dispersion is blind to lead-off
lies; the mean must carry them. Any σ-based detector needs the mean as a floor —
which is another reason mean+σ, not σ alone, is the only sensible form.

**F24f — harness integrity + the bug the scaffolding caught.** Identity family is
an EXACT no-op (max std 0, max|mean-single| 0). Mid-build, the smoke run inverted
(family AUROC .31): an early rerender re-derived each user instruction's location
from the FOLLOWING ack — but for a corrupted ack that is the planted WRONG
location, so it rewrote the instruction to agree with the lie and collapsed the
lie's surprisal 25.4 -> 0.0. Caught by dumping raw per-span values before trusting
the aggregate. Fixed by rewording assistant acks only, holding instructions
byte-identical. Recorded so the trap is not re-dug.

**Caveats.** (1) cleanFP not perfectly matched — fam .033 vs single .017 (n=20
calib floor noise) — inflates the catch delta somewhat. (2) n=30x2: catch
resolution ~.033/trace; treat sub-.03 deltas as noise. (3) family rewords ACKS
only (instructions held for faithfulness); a richer faithful family (instruction
paraphrase, or model-generated) might surface more dispersion — an under-powered
perturbation is not excluded (cf F20d). (4) k=6 against a 7-template pool → members
repeat, understating dispersion.

**Verdict.** The dual-representation idea earned a real but modest confirmation
(dispersion carries signal; family > instance). The practical detector gain is
mostly variance-reduction-by-averaging, not a new axis. The load-bearing "distinct
new signal" claim is UNRESOLVED, gated by task ceiling — the honest next move is a
harder task with headroom, not a bigger claim. Files: `loop/paraphrase.py`,
`loop/f24_murmuration.py`, `tests/test_paraphrase.py` (8 CPU tests).

## F23 — Ensemble-surprisal RESULTS: H-ens-2 passes, but sigma (not member
## count) does the work — the eBP framing is only weakly supported
## (2026-07-21; Qwen3-1.7B, entity_prose, 60 traces x 3 seeds, GPU).

    arm            auroc    Δauroc   catch   Δcatch  cleanFP  memberSD  wall
    single         0.9768   +0.0000  0.733   +0.000  0.000    0.0000     24s
    control-k4-s0  0.9768   +0.0000  0.733   +0.000  0.000    0.0000    104s
    k4-s0.005      0.9771   +0.0003  0.739   +0.006  0.000    0.098     249s
    k4-s0.01       0.9770   +0.0002  0.767   +0.033  0.006    0.171     248s
    k4-s0.02       0.9775   +0.0007  0.767   +0.033  0.006    0.308     249s
    k4-s0.05       0.9822   +0.0054  0.828   +0.094  0.017    0.675     250s
    k8-s0.01       0.9776   +0.0008  0.750   +0.017  0.000    0.193     499s
    k8-s0.02       0.9785   +0.0017  0.767   +0.033  0.006    0.357     499s
    k4-s0.02-r64   0.9762   -0.0005  0.728   -0.006  0.000    0.329     256s

    GATE: control PASS · H-ens-1 FAIL (+.0054<.02) · H-ens-2 PASS
          (+.094 @ cleanFP .017) · H-ens-3 PASS (|Δ|rank8-64 = .0013)

**F23a — The headline is true but thin: the ensemble DOES recover missed lies
(H-ens-2 PASS).** Best arm k4-s0.05 lifts catch .733 -> .828 (+.094, clears the
+.05 bar) while clean false-excision stays .017 (<= the .02 ceiling), and AUROC
rises +.0054 — a THRESHOLD-FREE confirmation that separability genuinely improved,
not merely a floor shift. In absolute terms it recovers ~9.5 of the ~26.7 points
the single-pass floor was missing (~1/3 of the gap). Real, cheap, modest.

**F23b — H-ens-1 FAILS exactly as pre-registered (F22 amendment was right).**
Best Δauroc +.0054 vs a +.02 bar. Baseline AUROC is already .977 (F21e flag hit
.993) — there is almost no ranking headroom, so a null here was expected and is
not evidence against the method. The load-bearing axis was always the absolute
floor (H-ens-2), and that is where the gain showed up.

**F23c — THE HONEST FINDING: perturbation MAGNITUDE (sigma), not member count
(k), is the active ingredient — which partly UNDERCUTS the eBP framing that
motivated the test.** eBP's mechanism is variance-reduction-by-averaging: more
members -> tighter estimate. The data show member count doing almost nothing and
sigma doing everything:

- k does the eBP-predicted thing, but WEAKLY: k4->k8 improves AUROC by ~+.001
  (s0.01: .9770->.9776; s0.02: .9775->.9785) and does NOT improve catch (s0.01:
  .767->.750, worse). Consistent with mild variance reduction, an order of
  magnitude too small to matter.
- sigma does everything: the +.05 bar is cleared ONLY at s0.05, and the AUROC
  and catch both scale monotonically with sigma, not k.

So the effect is mostly "a large low-rank weight perturbation stresses fluent
clean spans more than already-surprising lie spans, widening the gap," NOT
"averaging over an ensemble recovers the true marginal." That is a DIFFERENT
mechanism from the one Pitkow motivated. H-ens-2 passing is not eBP vindication.

**F23d — The confound to resolve next (owed before any strong claim): is the
winner an ENSEMBLE effect or a single-perturbed-pass effect?** Every sigma-sweep
point used k=4; the winner s0.05 was tested ONLY at k=4; and the one clean k
comparison (s0.01/s0.02) shows k nearly inert. Strong implication: **k=1 at
s0.05 likely captures most of the +.094 at 1/4 the cost.** If it does, "ensemble"
collapses to "score once under a big perturbation" — cheaper, and decisively
non-eBP. `f22_ensemble.py` needs a k1-s0.05 arm (and k2/k4/k8 at s0.05 to see if
averaging adds anything at the operating sigma). Until then, do not describe F23
as an ensemble result; describe it as a perturbation result.

**F23e — H-ens-3 PASS: low rank suffices (this part IS Pitkow-consistent).** At
s0.02, rank-64 matched rank-8 to |Δauroc|=.0013 — and was in fact marginally
worse on catch (.728 vs .767). Their Σθ-is-low-rank finding carries: no reason to
pay for high-rank perturbations.

**F23f — Harness integrity held on the real model.** control-k4-s0 reproduced
single-pass to 4 decimals with member SD exactly 0.0 (sigma=0 is a true no-op),
so every non-zero delta above is signal, not plumbing. The rank-64 arm going
slightly NEGATIVE (-.0005 auroc, -.006 catch) is a useful reality check: the
pipeline is not rigged to manufacture gains.

**Caveats.** (1) s0.05 is a LARGE perturbation (~5% Frobenius/layer) — the
"member" is a substantially degraded model, straining the eBP analogy; acceptable
only because we SCORE, never generate, from it. (2) cleanFP .017 sits close to the
.02 ceiling and the s0.05 floor dropped far (18.90 -> 13.45), so the operating
point is sensitive; AUROC (+.0054) is the trustworthy number, and it is modest.
(3) Numbers are 3-seed means; per-seed spread not yet inspected. (4) Cost: perturbed
arms ran ~250s (k4) / ~500s (k8) vs 24s single — the forward-hook overhead (196
callbacks/pass) is real; a k1-s0.05 confirmation would also be the performant path.

Run: `python -m source_monitor.llm.loop.f22_ensemble`; raw in
`results/llm_f23_ensemble_results.jsonl`, log in `results/f23_sweep.log`.

## F22 — Unbelievable marginals: a theoretical account of F20a, and a
## pre-registered ensemble test (2026-07-21). THEORY + DESIGN — NO RESULTS YET.

Source: Pitkow, Ahmadian & Miller, _Learning unbelievable probabilities_, Adv
Neural Inf Process Syst 24:738–746 (2011). Read into the project 2026-07-21.

**This entry contains no measurements.** It is a framing plus three
pre-registered hypotheses. Results land in F23. Do not cite F22 as evidence.

**F22a — There exist inference targets that NO parameter setting can reach.**
Stable fixed points of loopy belief propagation are minima of the Bethe free
energy, so a target is reachable only where the Bethe Hessian is positive-definite
there. Pitkow et al. prove many ordinary marginals fail that test — minimal
example: 4 binary nodes, uniform pairwise coupling, J > 0.316 — and name them
*unbelievable*: BP cannot converge to them **for any parameters**, so every
learning algorithm aimed at them necessarily fails, sometimes landing worse than
no learning at all. Not a corner case: for 8-node fully-connected Ising models
with random couplings, most targets go unbelievable once coupling is moderate.

**F22b — That is F20a's structure, stated as a theorem in a different system.**
F20a found LoRA drives competence .88->1.0 and detection ->1.0 while bsi holds at
~.21; F20e found the repair leg is excise-and-regenerate at INFERENCE, not any
training intervention. Pitkow et al. supply the clean reason a system can behave
that way: when an approximate inference procedure's REACHABLE SET excludes the
correct answer, parameter search is the wrong search. This is the strongest
theoretical argument the project has for why SEER is an inference-time loop
rather than a fine-tune.

**STATED LIMIT — do not overclaim.** Transformers do not run loopy BP. The
theorem does not transfer as a theorem, and no result in this repo depends on it.
What transfers is an existence proof for a CATEGORY — errors that are not training
errors — plus evidence the category is common. F22 is framing and a hypothesis
source, never evidence.

**F22c — Their remedy is an ensemble, which suggests a detector upgrade.** When
the target is unbelievable, Bethe wake-sleep learning never settles; parameters
keep moving. But at equilibrium the TIME AVERAGE of the BP fixed points equals the
unbelievable target exactly (their Thm 2) — even though no individual fixed point
need be near it. Averaging beliefs over an ensemble of perturbed parameters
reaches what no single parameter setting can. Two details matter for us: (i) a
gaussian ensemble θ ~ N(θ̄, Σθ) works nearly as well as continued learning, and
(ii) Σθ was consistently LOW-RANK — one or two principal components sufficed.

Our detector reads surprisal off ONE forward pass at ONE parameter setting, i.e.
one fixed point, and the F21e floor is calibrated in nats against that single
estimate. If the estimate carries fixed-point-specific noise, averaging over a
low-rank perturbation ensemble should tighten it.

**Pre-registered (test in F23, `loop/f22_ensemble.py`):**

- **H-ens-1 (separability).** Ensemble-averaged span surprisal separates planted
  lies from genuine self-spans better than single-pass: AUROC gain >= +.02.
- **H-ens-2 (the 19%).** At matched clean false-excision rate (<= .02), ensemble
  scoring raises the planted-lie catch rate over single-pass by >= +.05,
  recovering part of the ~19% of lies the q=.99 floor currently misses (F21e).
- **H-ens-3 (low-rank suffices).** Rank r <= 8 perturbation performs within .01
  AUROC of a much larger rank — their Σθ finding carried over.

Control: sigma=0 must reproduce single-pass EXACTLY (teacher-forced scoring is
deterministic). Any measured gain at sigma=0 is a harness bug, not a result.

**AMENDMENT, before any full run (learned from F21b — check the ceiling FIRST).**
H-ens-1 is probably near-saturated by construction and will be weak evidence
either way: F21e already reports flag hit rate .993, i.e. the argmax picks the
corrupt span ~99% of the time, so span-level AUROC has almost no headroom. The
load-bearing test is **H-ens-2** — the binding constraint is the ABSOLUTE FLOOR
(the ~19% of lies q=.99 misses), not the ranking. Report H-ens-1 for completeness;
do not treat a null there as a verdict on the ensemble.

**Harness verified 2026-07-21 (`--quick`, 6 traces, 1 seed, Qwen3-1.7B).**
Control PASSES: sigma=0 reproduces single-pass bit-exactly (Δauroc = 0.0e+00,
member SD = 0.0e+00). The perturbation is live at sigma=.01 (member SD = .215
nats, i.e. members genuinely disagree). Held-out floor came out 16.09 nats
against F21e's 16.26 — different quantile convention (per-trace max at q=.98 here
vs per-span q=.99 there), so the agreement is a sanity check, not a match. AUROC
pinned at 1.000 in all arms, as expected at n=6: no headroom, no information.
Smoke test only — plumbing, not evidence.

**Methodology note.** First draft calibrated the floor on the same clean traces it
then measured the false-excision rate on, which makes that rate true by
construction and fits the floor to the eval data. Corrected to a held-out
calibration split (`calib_seed=7`, disjoint from eval seeds), matching F21e.

**What does NOT import: eBP as a repair mechanism.** eBP assumes the true
marginals are perpetually available during learning — supervision we do not have
at inference. The gaussian-ensemble variant needs Σθ, which needs that supervision
once. Take the DETECTOR idea; leave the repair as excise-and-regenerate (F20e).

**F22d — Latent-MoE assessed (`~/Repositories/Latent-MoE`): the idea imports, the
code does not.** LatentMoE (Elango et al., NVIDIA 2026, arXiv:2601.18089) is an
untrained `nn.Module` — a drop-in MoE FFN for a model you are PRETRAINING. It
cannot be grafted onto pretrained Qwen3-1.7B (dense) without training it, so it
supplies nothing to the F23 experiment. But the mechanism it optimizes is exactly
eBP's requirement: **top-k routing already IS an ensemble over parameter subsets**
— each token is processed by a different subset of expert weights — and LatentMoE
exists to expand that combination space (N'=αN, K'=αK). On an MoE checkpoint you
could draw eBP ensemble members by perturbing the ROUTER (temperature, sampling
instead of argmax top-k, dropping a slot) at zero weight-copy and near-zero memory
cost, which is strictly cheaper than the weight perturbation F23 uses.

- **H-ens-4 (router ensemble), NOT yet testable here.** Needs an MoE checkpoint
  that fits 12 GB. Qwen3-30B-A3B does not (~60 GB bf16). OLMoE-1B-7B (~7B total)
  is the plausible candidate, likely 8-bit. Deferred, not rejected. See F22e —
  a local GGUF MoE does NOT unlock this.

**F22e — Unsloth Studio's API cannot host the detector (probed 2026-07-21,
localhost:8889, `unsloth/Qwen-AgentWorld-35B-A3B-GGUF:UD-Q4_K_XL`, llama.cpp
build b10069).** Tested against the live endpoint, not the docs:

    /v1/chat/completions + logprobs   -> 400 "logprobs is not supported
                                          for chat completions"
    /v1/completions + echo:true       -> echo SILENTLY IGNORED. 7 prompt tokens
                                          in, 1 logprob entry out (the generated
                                          token only)
    /v1/completions + prompt_logprobs -> ignored (vLLM-ism), logprobs: null
    /completion, /tokenize, /props    -> 405, wrapper does not expose the
                                          native llama.cpp surface

The detector needs per-token logprobs for text ALREADY IN THE CONTEXT
(teacher-forced retrospective surprisal). This API only returns logprobs for
tokens it generates, so the answer is no. Scoring token-by-token via top-N
lookup is not a workaround: a planted lie's token is exactly the one that falls
outside the top-N, so the data would be censored precisely at the tail the
detector exists to measure.

**Revises the GGUF risk assessment from the previous session.** The stated worry
was logprob FIDELITY under quantization (the floor is calibrated in nats). The
actual blocker is one level upstream — the logprobs are not exposed at all.
Fidelity remains untested because it is unreachable from here.

**The GGUF path is open through llama-cpp-python — now VERIFIED, not asserted.**
The weights are already in the HF cache
(`models--unsloth--Qwen-AgentWorld-35B-A3B-GGUF`). `llama-cpp-python` 0.3.34 is
installed in the source-monitor venv, and `scripts/verify_gguf_logprobs.py`
confirms per-token logprobs for SUPPLIED text on a tiny CPU GGUF (stories260K),
by two independent paths:

- `create_completion(echo=True, logprobs=N)` returns one `token_logprob` per
  PROMPT token (first is `None` — nothing predicts it), i.e. the OpenAI-style
  surface that llama-server drops but the wrapper honours.
- `logits_all=True` + low-level `eval` gives the exact neg-logprob of an
  arbitrary supplied span — the direct analogue of `telemetry`'s teacher-forced
  scoring.

User-confirmed mechanism (matches the observed behaviour): llama-server skips
copying prompt-token logits to host during the parallel prefill for speed; the
Python wrapper forces per-token evaluation, so the logits survive. So the F22e
blocker is Unsloth's *server*, not GGUF. The quantization-fidelity question (does
the nats-calibrated floor survive Q4_K_XL noise?) is now actually runnable — it
needs a `telemetry`-parity GGUF scorer over `entity_prose`, on GPU with fans.

**F22f — optillm assessed (`~/Repositories/optillm`): same shape as SEER, opposite
goal; import the signals, not the server.** optillm is an OpenAI-compatible
inference-time-compute proxy (20+ methods: moa, mcts, best-of-n, plansearch,
rstar, self-consistency, cot_decoding, entropy_decoding). Architecturally it is
exactly SEER's shape — a loop around a frozen model — but its objective is the
mirror image: it spends inference compute to produce a BETTER answer, whereas
source-monitor spends it to CATCH a wrong one. That difference decides what is
worth taking.

- **Does NOT import: the scoring surface.** optillm's local `calculate_logprobs`
  (inference.py) is teacher-forced HF next-token logprob extraction — the same
  operation as `telemetry.retrospective_surprisal`, minus the span/slot
  provenance. It adds nothing to the detector and, being HF-only, does nothing
  for GGUF (for GGUF backends optillm proxies to an OpenAI endpoint and inherits
  the very echo/prompt-logprob limitation F22e hit). The 20+ accuracy methods are
  orthogonal to detection.
- **Imports: two candidate detector signals, complementary to surprisal.**
  (1) cot_decoding's confidence Δ = mean over answer tokens of (p_top1 − p_top2)
  — a MARGIN signal, not a neg-logprob; low margin = model torn. (2)
  entropy_decoding's entropy/varentropy (entropix-style). Both are cheap
  per-token quantities already computed in the forward pass. Worth trying as
  extra features once the detector is more than a single scalar.
- **Imports: a template for the CoT-monitor thread.** cot_decode branches over
  the top-k FIRST tokens, greedily decodes each path, scores each by answer
  confidence, and keeps/aggregates. That branch-score-select structure is
  literally the "score reasoning steps, excise the faulty one, re-reason"
  architecture parked as the next-strongest SEER application. optillm is a
  working reference for it.
- **Caveat — do not naively swap Δ in for surprisal.** cot_decoding's Δ is
  measured on tokens it GENERATES, not retrospectively on a claim already in the
  context. Using it to monitor an emitted span means re-generating that span,
  which changes what is being measured. Δ is a generation-time signal;
  retrospective surprisal is a scoring-time one. They are not interchangeable.

**And it does NOT unlock H-ens-4, despite being an MoE.** 35B-A3B is the right
shape, but router perturbation needs per-expert routing control in Python, which
llama.cpp does not expose either. MoE-over-GGUF gets us neither the scoring nor
the router. H-ens-4 still needs HF transformers (OLMoE-1B-7B).

Working as an agent endpoint, for the record: tool calling verified good
(`finish_reason: tool_calls`, well-formed arguments, 37.8 tok/s), plus
Anthropic-dialect `/v1/messages` and server-side python/web_search/terminal
tools. Fine for agent work; wrong instrument for this measurement.

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

**F21e — FIXED: an absolute calibrated floor makes the monitor quiet when idle
(P-3.3 PASS).** Replaced the relative within-trace z-rule with an ABSOLUTE floor =
the 99th percentile of genuine self-span scores on CLEAN traces held out from the
eval seeds (floor = 16.26 nats for Qwen3-1.7B / entity_prose).

    metric                 z-rule    absolute floor
    clean false excision    .956        .017
    clean accuracy          .961       1.000
    flag hit rate           .952        .993
    planted monitor_on      .961        .944

P-3.3 PASS (excise .017, zero accuracy cost); P-3.2 PASS harder (hit .993, within
.034 of oracle). P-3.1 = +.111 against an oracle ceiling of +.145, i.e. 77% of the
achievable repair (the +.15 bar remains above the ceiling — see F21b).

The trade is explicit and tunable: q=.99 is conservative, so the floor misses ~19%
of lies (planted excise .917 -> .811), which is why monitor_on dips slightly.
`calib_quantile` is the precision/recall knob — lower it to catch more lies at the
cost of more idle excisions. Both the MECHANISM and the TRIGGER POLICY are now
validated; a relative outlier test is the wrong policy at small span counts because
"most surprising span" is not "actually surprising span".

**Next:** (1) span SEGMENTATION for free-form text — the one real generality gap
(the templated task gave us "one claim per turn" for free; sentence/clause
splitting is the v1); (2) verify logprob fidelity under GGUF/quantization and
recalibrate the floor per quant level (excision there is a context edit, which is
strictly easier than our attention-mask); (3) monitor the model's own REASONING
(<think> blocks) rather than only its final claims; (4) long-ξ task; (5) 4B.

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
