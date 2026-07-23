# source-monitor — build handoff (F27+)

**Audience:** a coding agent (or engineer) with **no prior context** on this repo.
Read §0 and §1 fully before writing any code. §1 is not optional — every trap
listed there has already cost a wasted run in this project.

**Date:** 2026-07-21. **Companion docs:** `FINDINGS.md` (the experiment log,
F0–F26d — the source of truth for what is known), `CONCEPTUAL-NOTES.md` (the
"why", optional). **Status of the work:** both detector legs of the monitor are
validated in principle; what remains is the router that combines them, a harder
evaluation task, and the chain-of-thought extension.

---

## 0. Orientation (what this project is)

seer/source-monitor builds an **inference-time monitor** around a *frozen,
competent* LLM. It does not retrain the model. The loop is: **detect** a
self-generated error → **excise** the offending span from context → **regenerate**.
The blind spot being fixed (established earlier, F20a): ordinary fine-tuning fixes
competence but not the model's tendency to build on its own confident falsehoods;
the repair has to happen at inference (F20e).

Two detector **legs** now exist, and they use **different instruments**:

| Leg | Detects | Instrument | Works when | Files |
|-----|---------|-----------|-----------|-------|
| 1. Surprisal | context-**contradicted** errors | teacher-forced retrospective surprisal (value-slot) | the truth is present in context (F21e, F25) | `llm/telemetry.py`, `llm/loop/monitor.py` |
| 2. Consistency | **factual** confabulation | k sampled generations at temperature, measure agreement | the truth is NOT in context / pure recall (F26d) | `llm/loop/consistency.py`, `llm/loop/f26_sample.py` |

Leg 1 is near-perfect where it applies (single-pass AUROC ~.99). Leg 2 gets
AUROC ~.76–.82 separating known from unanswerable. The **key result that motivates
everything below (F25/F26):** surprisal ceilings whenever the lie contradicts
context, so it needs no booster there; the open frontier is errors that are *not*
context-contradicted (factual hallucination, self-consistent reasoning slips),
which is exactly where surprisal is blind and leg 2 (or the CoT scorer) is needed.

**Stack:** Python 3.13, `uv` venv at `.venv/`, torch 2.13 (cu130), model
`Qwen/Qwen3-1.7B` loaded via `source_monitor.llm.cache.load_model`. Config in
`llm/loop/config.py` (`Phase3Config`). Tests: `pytest tests/` (92 pass as of
this writing). Results land in `results/*.jsonl`. Run a module with
`.venv/bin/python -m source_monitor.llm.loop.<name>`.

**GPU:** RTX 3060 12 GB, broken fan — it runs hot but **holds stably at ~93 °C
with floor fans on** (owner-confirmed; do not halt for temperature alone). For any
run >~2 min, launch in the background (`nohup … &`) and poll with **short** shell
calls — long-blocking calls drop the connection (the detached job survives).
Monitor with `nvidia-smi`.

---

## 1. Known traps — DO NOT RE-DIG

Every one of these already happened. The value in this project has been *catching
artifacts*, not writing code; the modules are small. Reproduce these and you will
get confident, wrong numbers.

1. **Faithfulness invariant** (F24f). When you build any perturbation/paraphrase/
   sampling **family** of a trace, hold the **facts** and the **scored span**
   byte-identical; only vary what you intend to vary. Never derive a fact from a
   field that can be corrupted. *The bug:* an early family generator re-rendered a
   user instruction's location from the *following* assistant turn — which for a
   planted-lie turn was the WRONG location — so it rewrote the instruction to agree
   with the lie, erased the inconsistency, and **inverted the detector** (AUROC
   .31). Guard: an **identity family must reproduce single-pass scoring exactly**
   (dispersion 0.0). If it doesn't, stop — it's a harness bug.

2. **Negation / abstention prior bias** (F26a). Whole-claim `mean_neglogp` is
   length-biased toward long fluent hedges ("I have no reliable record of that.")
   over short answers ("Paris."), and a system prompt that primes abstention makes
   the hedge "preferred" for *every* question. Use **value-slot** scoring
   (`ClaimScore.value_only_neglogp`) and, when picking the model's preferred
   answer, compare **value-surface candidates only** (exclude negation). This is
   the repo's Phase-0b negation-prior bias resurfacing; expect it anywhere you
   compare candidates of different lengths/registers.

3. **Check the ceiling before trusting a bar** (F21b, F25). Multiple "FAIL"s in
   this log were *pre-registration errors* — the bar was set above the task's
   achievable ceiling (the oracle). Multiple "small effects" were real but small
   *because the single-pass baseline was already near-perfect*. Before claiming a
   pass/fail: compute the oracle/realistic ceiling, and check whether single-pass
   already saturates the task. Report deltas against the realistic ceiling.

4. **Every detector run needs a no-op control** that must reproduce single-pass
   *exactly* (the identity family in F24, `sigma=0` in F23). If the control drifts,
   nothing downstream is trustworthy.

5. **Look at raw values when an aggregate looks off.** Both major bugs were caught
   by dumping per-span / per-sample numbers, not by reading the AUROC summary. An
   inverted AUROC or a degenerate stability (everything = 1.0) is a signal to
   inspect rows, not to interpret the mean.

6. **Statistical honesty.** Runs are n≈30–60; catch-rate resolution is ~1/n per
   trace. Treat any delta smaller than ~1 trace as noise. Firm real claims with
   multiple seeds before writing them into FINDINGS.

---

## 2. Build A — the Router  *(agent-safe mechanism; derivability classifier needs judgment)*

**Goal.** Per assistant span, decide *"is this claim's truth derivable from the
context?"* and dispatch: derivable → Leg 1 (surprisal); not-derivable/factual →
Leg 2 (sampled consistency). Then combine into one per-span flag. This is the
missing piece that turns two validated legs into one monitor.

**Precision-weighting** (design refinement, from the error-neuroscience analogy).
The maximum-danger cell is **high model confidence × context-underivable** — a
confidently-stated fact with no in-context check. Route by *confidence ×
derivability*, and have Leg 2 fire hardest exactly there.

**Concrete steps.**
1. `llm/loop/router.py`: `route(trace, span) -> "surprisal" | "consistency"`.
   v1 derivability heuristic: is the claim's value present in, or entailed by, the
   prior context turns? (For `entity_prose`/`arithmetic` the answer is in context →
   surprisal; for `factual_qa` grounded=False it is not → consistency.) A trivial
   robust v1: check whether the claim value string (or a simple derivation of it)
   appears in the context; **this classifier design is the NEEDS-JUDGMENT part** —
   keep it simple and be explicit about its failure modes.
2. `combined_flag`: run the routed leg, produce a calibrated flag (reuse Leg 1's
   absolute-floor calibration for the surprisal branch; Leg 2's agreement/
   distinct-ratio threshold for the consistency branch).
3. **Pre-register:** on a MIXED eval (e.g. `entity_prose` + `factual_qa` combined),
   the routed monitor's catch-rate ≥ max(leg1-only, leg2-only) at matched clean
   false-flag rate. **Control:** an all-context-derivable eval must make the router
   ≡ Leg 1 exactly; an all-factual eval ≡ Leg 2 exactly (the trap-1 discipline,
   applied to routing).

**Reuse:** `monitor.span_scores` + `monitor.calibrate_floor` (Leg 1);
`consistency.sampled_consistency` (Leg 2); `f22_ensemble.auroc`, `_evaluate` for
scoring. Mirror the runner shape of `f26_sample.py`.

---

## 3. Build B — a harder confabulation task  *(NEEDS-JUDGMENT — do with the reviewer, not cold)*

**Why.** Leg 2's win (F26d) came mostly from **hedge-rate**: `factual_qa`'s
unanswerable questions are *obviously* unanswerable ("grains of sand on Brighton
beach"), so Qwen3 correctly abstains. That is calibrated refusal, not the sampling-
variance signal (`distinct_ratio`) we actually want to validate. The dangerous
case — the model **confidently confabulating a plausible-but-wrong answer that
varies across samples** — is under-represented, so `distinct_ratio` has not been
stress-tested.

**Goal.** A task of *plausible-but-obscure* questions the model half-knows (obscure
real entities, fine-grained facts) so it confabulates *varying* confident answers
rather than abstaining. Ground truth known. Success = `distinct_ratio` separates
correct-and-known from confabulated at AUROC well above chance *without leaning on
hedge-rate*.

**Why not a cold agent:** designing a task that (a) defeats single-pass surprisal
AND (b) triggers confident confabulation is an *empirical* design problem — the
same one that made F25's "hard task" ceiling out. It needs iteration and judgment
about what Qwen3-1.7B actually does. Draft it, then review the raw sampled answers
before trusting any AUROC.

---

## 4. Build C — CoT non-causal step scorer  *(mechanism agent-safe; substrate needs judgment)*

**The idea** (from Merrill/Sabharwal-style CoT-expressivity + the "First-Principles
Theory of Slow Thinking" paper, distilled in `CONCEPTUAL-NOTES.md`). Our surprisal
detector is **causal** — it scores each token on its prefix only — which is
provably blind to self-consistent errors. The fix is **explanatory (non-causal)
scoring**: evaluate each reasoning step *in hindsight*, conditioned on the **whole
chain including later steps and the final answer** ("given where this ended up,
which step doesn't cohere?"). Theorem 3.1 of that paper proves the causal view
cannot even approximate the right (posterior) one; Remark 3.12 confirms the
non-causal read is inference-legal (do it in the prefill pass).

**Mechanism (agent-safe).** Given a reasoning trace segmented into steps
`s_1..s_m` and a final answer:
- Causal score of step i: mean neglogp of `s_i`'s tokens given `s_1..s_{i-1}` (what
  we already do).
- **Non-causal score** of step i: mean neglogp of `s_i`'s tokens given *all other
  steps + the answer* (re-encode the full trace, score `s_i` in place). A practical
  proxy on a causal LM: score `s_i` with the answer + subsequent steps **prepended/
  appended** into the context so they're visible during `s_i`'s scoring.
- **Pre-register:** the non-causal score locates the first faulty step better than
  the causal score (AUROC_noncausal > AUROC_causal for "is this the wrong step").

**Substrate (NEEDS-JUDGMENT).** You need a task where Qwen3 **generates groundable
flawed reasoning** — multi-step problems with per-step Python ground truth, and the
model actually makes catchable self-consistent slips. Turn thinking **on** (the
loop currently strips `<think>` — see `monitor.THINK_OFF`/`strip_think`; you'll
reverse that here), segment the emitted reasoning into steps. This is the same
hard-substrate problem as Build B; do not expect a cold agent to design it well.

---

## 5. Reviewer protocol (how PRs get checked)

Review for **artifact-vs-signal**, not just "it runs". For every result PR, confirm:
1. the **identity/no-op control** reproduces single-pass exactly;
2. **raw per-span/per-sample values** were inspected and look sane (no inversion,
   no degeneracy);
3. the result is compared against the **realistic ceiling**, not an arbitrary bar;
4. the **faithfulness invariant** held (scored span + facts byte-identical across
   the family);
5. deltas larger than **~1-trace noise**, ideally multi-seed;
6. new findings appended to `FINDINGS.md` in the existing F-numbered style, with
   the honest caveats stated (this repo's convention is to write down what *didn't*
   work and why, not just the wins).

---

## 6. Repo map & commands

```
src/source_monitor/llm/
  cache.py            load_model(name, device, dtype, enable_thinking)
  telemetry.py        retrospective_surprisal, ClaimScore aggregations (Leg 1 core)
  provenance.py       tokenize_with_provenance, span annotations
  ood/                task generators (all yield OODTrace w/ .turns, .claim, .meta)
    entity_prose.py   short-ξ entity tracking (corrupt_mid plants a mid lie)
    arithmetic.py     long-ξ running total (corrupt_mid = off-by-small; F25)
    factual_qa.py     recall QA, grounded flag; base for Leg 2 (F26)
    base.py           OODTrace/OODClaim, make_variant, raw_claim_score
  loop/
    config.py         Phase3Config (model, seeds, calib, thresholds)
    monitor.py        build_context, span_scores, flag_index, calibrate_floor,
                      holed_mask, generate_answer, THINK_OFF/strip_think (Leg 1 loop)
    ensemble.py       low-rank weight perturbation (F23)
    f22_ensemble.py   auroc, _evaluate  ← REUSE these for any new eval
    paraphrase.py     faithful family generators (F24; pluggable rerender_fn)
    f24_murmuration.py
    consistency.py    Leg 2: teacher-forced (dead end) + sampled_consistency (works)
    f26_sample.py     Leg 2 viability runner  ← template for new runners
tests/                pytest; every family/scorer gets CPU faithfulness tests
FINDINGS.md           the log — READ IT
results/*.jsonl       run outputs (gitignored)
```

Run examples:
```
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m source_monitor.llm.loop.f26_sample --quick        # Leg 2 smoke
.venv/bin/python -m source_monitor.llm.loop.f24_murmuration --quick   # F24 smoke
```

**First task for a new agent:** read `FINDINGS.md` F25 + F26 (the boundary and the
two legs), run the two smokes above to confirm the environment, then take Build A.
