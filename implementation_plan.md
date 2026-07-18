# Phase 0 — Does Retrospective Surprisal Port to an LLM?

Port source-monitor's zero-shot detection signal (R3: AUROC .93–1.00 on all corruption types) from the 4.9M-param toy decoder to Qwen3-1.7B. No training — pure inference. The first rung of the integration plan's ladder.

## Core Question

> At each emission, the LLM computes the ingredient natively: when it samples a token, it has that token's log probability under its own predictive state. Does retrospective re-scoring of self-generated spans catch planted errors at AUROC ≥ .9, zero-shot?

## User Review Required

> [!IMPORTANT]
> **Model choice: Qwen3-1.7B vs 0.6B.** The integration doc calls for 1.7B. On the 3060 (12GB), bf16 inference at 1.7B is ~3.4GB — comfortable. However, the seer repo already has a pinned 0.6B snapshot (`Qwen/Qwen3-0.6B` at `c1899de2`). I'll implement the harness to accept any Qwen3 checkpoint via config, and the Phase 0 experiment will target 1.7B as the doc specifies. **Does this seem right, or would you prefer to start with 0.6B for faster iteration and step up?**
>
> **→ RESOLVED (Fable review): both, sequentially.** Debug the harness on 0.6B
> (cached, fast), then run BOTH 0.6B and 1.7B as the reported experiment —
> inference is <1 min each, and the size comparison is free signal about how
> detection scales. Use the post-trained (chat) checkpoints, not `-Base`: the
> task is chat-format and the emissions are assistant turns.

> [!IMPORTANT]
> **Location.** The integration doc says "New LLM-scale work lives HERE (a future `llm/` subpackage), self-contained as always." I'll create `src/source_monitor/llm/` in the source-monitor repo. The seer repo stays untouched as the design archive. **Confirm this is the right home.**
>
> **→ RESOLVED: confirmed.** `src/source_monitor/llm/`, reusing `metrics.py`
> only. seer stays archival per SEER-INTEGRATION.md §0.

> [!WARNING]
> **New dependency: `transformers>=4.51`.** Loading Qwen3 requires the `transformers` library (and `huggingface_hub` for cache resolution). I'll add these to `pyproject.toml` under a new `[project.optional-dependencies] llm` extra so the toy-scale code stays zero-dependency beyond torch+numpy.
>
> **→ RESOLVED: approved as an `llm` extra.** `accelerate` is not strictly
> needed for single-GPU `.to("cuda")` bf16 but is harmless; keep. Note for
> later phases: Gemma 4 E-series will likely need a newer transformers floor
> than Qwen3 does — pin per-phase, not globally.

## Open Questions

1. **Task complexity.** The toy uses 8 ops × 4 objects × 3 containers. For the LLM rendered to natural text, do you want the same parameters or something larger to stress the model's tracking?
   **→ RESOLVED: keep 8×4×3 as primary (comparability with the toy anchors),
   and add one harder config (12×6×4) as a secondary row — if detection is
   at ceiling on the easy config, the hard config shows where it bends.**
2. **Number of traces.** The toy ran 400 eval traces per arm. I'll default to 200 traces (× 3 corruption types = 600 corrupted + 200 clean = 800 forward passes). With bf16 on the 3060, each pass over a ~200-token context takes < 50ms, so the full experiment is < 1 minute of GPU time. Bump up or down?
   **→ RESOLVED: 400.** Inference is nearly free here; the stratified
   analyses in Amendment A2 want the headroom.
3. **Multi-seed?** Phase 0 has no training, so the only stochasticity is task generation. I'll run 3 seeds for the task RNG and report mean ± std.
   **→ RESOLVED: yes, 3 task-RNG seeds as planned.**

---

## Proposed Changes

### LLM Subpackage Skeleton

New `src/source_monitor/llm/` subpackage — self-contained, no imports from the toy model code (reuses `metrics.py` only).

---

#### [NEW] [\_\_init\_\_.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/__init__.py)

Package init with version marker.

---

#### [NEW] [config.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/config.py)

`LLMExperimentConfig` — typed configuration for Phase 0:
- `model_name: str` (default `"Qwen/Qwen3-1.7B"`)
- `device: str` (default `"cuda"`)
- `dtype: str` (default `"bfloat16"`)
- Task parameters: `n_ops`, `n_objects`, `n_containers`, `remove_prob`
- Experiment parameters: `n_traces`, `seeds`, `corruption_types`
- Aggregation method: `"mean"` or `"min"` logp over span

---

#### [NEW] [cache.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/cache.py)

Model loading — fail-closed local-only resolution (adapted from seer's pattern, stripped to essentials):
- `load_model(model_name, device, dtype)` → `(model, tokenizer)` — loads from HF cache, bf16, `eval()` mode, no gradient
- No network fallback: if the snapshot isn't cached locally, error with a download command

---

#### [NEW] [task_render.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/task_render.py)

Convert the toy's entity-tracking task to multi-turn natural language.

```python
@dataclass
class Turn:
    role: str               # "system" | "user" | "assistant"
    content: str
    is_self: bool           # True for assistant turns (model's own emissions)
    step_index: int | None  # which operation step this corresponds to
    is_corrupted: bool      # set by corruption injection

@dataclass
class Trace:
    turns: list[Turn]
    query_object: str       # "the red ball"
    ground_truth_final: str # "box C" or "nowhere"
```

Key functions:
- `render_trace(task: Task) → Trace` — deterministic text rendering using object/container name maps
- `OBJECT_NAMES` / `CONTAINER_NAMES` — fixed name lists (e.g., "red ball", "blue cube", ...; "box A", "box B", ...)
- System prompt: concise instruction to track objects and report locations after each operation

Each operation step generates a user turn + assistant turn:
```
User: "Put the red ball in box A."
Assistant: "The red ball is in box A."
```

The assistant turn content is the model's "self-emission" — the text whose fidelity we'll test.

---

#### [NEW] [corruption.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/corruption.py)

Port the three corruption types to text-level span replacement:

```python
@dataclass
class CorruptionRecord:
    corruption_type: str          # "ghost" | "mislocation" | "phantom"
    step_index: int               # which step was corrupted
    original_content: str         # the genuine assistant response
    corrupted_content: str        # the planted false response
    trace: Trace                  # the full trace with corruption applied
```

Three injectors (same logic as [task.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/task.py#L184-L232), text-level):
- `inject_ghost_text(trace, rng)` — after REMOVE step, assistant says object is in a container
- `inject_mislocation_text(trace, rng)` — object present, assistant names wrong container
- `inject_phantom_text(trace, rng)` — object present, assistant says "nowhere"

Each returns a `CorruptionRecord` or `None` (if the trace has no eligible step).

---

#### [NEW] [provenance.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/provenance.py)

Layer 0 from the integration plan: provenance bookkeeping.

```python
@dataclass
class SpanAnnotation:
    start_token: int    # inclusive token index in the full sequence
    end_token: int      # exclusive token index
    kind: str           # "system" | "user" | "assistant"
    step_index: int | None
    is_corrupted: bool

def tokenize_with_provenance(
    tokenizer, trace: Trace
) -> tuple[Tensor, list[SpanAnnotation]]:
    """Tokenize a Trace using the model's chat template, tracking which
    token ranges are self-emitted (assistant) vs external (user/system)."""
```

This builds the token-level self/external mask that the telemetry module needs. Uses the tokenizer's `apply_chat_template` to get exact token boundaries.

---

#### [NEW] [telemetry.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/telemetry.py)

Layer 1 from the integration plan: the detection signal.

**Retrospective surprisal** (the key Phase 0 measurement):
```python
@torch.no_grad()
def retrospective_surprisal(
    model, input_ids: Tensor, self_spans: list[SpanAnnotation],
    aggregation: str = "mean",
) -> list[SpanScore]:
    """Teacher-forced forward pass over context. For each self-span,
    aggregate -logp of its tokens given the preceding context.
    
    Returns one SpanScore per assistant turn, with:
      - span_mean_neglogp: mean(-logp) over tokens in the span
      - span_min_logp: min(logp) over tokens in the span (most surprising token)
      - is_corrupted: ground truth label for AUROC
    """
```

This is the direct port of `_surprise_gamma` from [model.py L280-293](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/model.py#L280-L293), but simpler: the LLM's forward pass natively gives logits at every position, so we just read `log_softmax(logits[t-1])[token[t]]` for each self-token `t`.

**Emission-time surprisal** (secondary, cheaper):
```python
def emission_time_surprisal(
    model, tokenizer, prompt: str, 
    max_new_tokens: int,
) -> list[tuple[int, float]]:
    """Generate tokens and record -logp of each as it is sampled."""
```

For Phase 0 we focus on retrospective surprisal (the ghost-catcher, R3's true analog). Emission-time is implemented for completeness but the experiment uses retrospective.

---

#### [NEW] [phase0.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/phase0.py)

The Phase 0 experiment orchestrator:

```python
def run_phase0(config: LLMExperimentConfig) -> Phase0Results:
    """
    1. Load Qwen3-1.7B (bf16, eval, no grad)
    2. Generate n_traces entity-tracking tasks
    3. Render to multi-turn chat Traces
    4. For each trace, create clean + 3 corruption variants
    5. Tokenize with provenance
    6. Run retrospective surprisal over all self-spans
    7. Compute per-corruption-type AUROC
    8. Save results to results/llm_phase0_results.jsonl
    """
```

Output structure:
```python
@dataclass
class Phase0Results:
    model_name: str
    n_traces: int
    seed: int
    per_type_auroc: dict[str, float]       # ghost, misloc, phantom
    pooled_auroc: float                    # all types combined
    genuine_neglogp: list[float]           # distribution of genuine span scores
    corrupted_neglogp: dict[str, list[float]]  # per-type corrupt scores
    wall_seconds: float
```

**Success criterion (from SEER-INTEGRATION.md):** AUROC ≥ .9 zero-shot. If this fails, stop and understand why before any training.

---

### Tests

#### [NEW] [test_task_render.py](file:///home/ty/Repositories/ai_workspace/source-monitor/tests/test_task_render.py)

- `test_render_produces_valid_turns` — correct role sequence (system, then alternating user/assistant)
- `test_render_deterministic` — same seed → same text
- `test_self_positions_are_assistant_turns` — `is_self=True` only on assistant turns
- `test_object_container_names_consistent` — names in assistant turn match the operation

#### [NEW] [test_corruption_text.py](file:///home/ty/Repositories/ai_workspace/source-monitor/tests/test_corruption_text.py)

- `test_ghost_corrupts_removal_step` — after REMOVE, assistant says object is in a container
- `test_mislocation_changes_container` — container differs from ground truth
- `test_phantom_says_nowhere` — present object reported as nowhere
- `test_corruption_preserves_other_turns` — only one assistant turn is modified
- `test_returns_none_when_ineligible` — no valid step → `None`

#### [NEW] [test_provenance.py](file:///home/ty/Repositories/ai_workspace/source-monitor/tests/test_provenance_llm.py)

- `test_tokenize_with_provenance_spans_cover_all_tokens` — no gaps
- `test_assistant_spans_marked_self` — correct kind assignment
- `test_round_trip_with_fake_tokenizer` — mock tokenizer for CPU-only testing

#### [NEW] [test_telemetry.py](file:///home/ty/Repositories/ai_workspace/source-monitor/tests/test_telemetry_llm.py)

- `test_retrospective_surprisal_shape` — one score per self-span
- `test_high_logp_for_predictable_tokens` — on a trivially predictable sequence, logp ≈ 0
- `test_low_logp_for_random_tokens` — random insertions get high -logp
- Uses a tiny randomly-initialized model (no download) for deterministic testing

---

### Dependency Changes

#### [MODIFY] [pyproject.toml](file:///home/ty/Repositories/ai_workspace/source-monitor/pyproject.toml)

Add optional dependency group:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
llm = [
    "transformers>=4.51",
    "huggingface_hub>=0.25",
    "accelerate>=1.0",
]
```

This keeps the toy-scale code (`pip install source-monitor`) dependency-free beyond torch+numpy. The LLM experiments are opt-in via `pip install source-monitor[llm]` or `uv sync --extra llm`.

---

## File Summary

| File | Type | Purpose |
|------|------|---------|
| `src/source_monitor/llm/__init__.py` | NEW | Subpackage init |
| `src/source_monitor/llm/config.py` | NEW | Typed experiment configuration |
| `src/source_monitor/llm/cache.py` | NEW | Fail-closed model loading |
| `src/source_monitor/llm/task_render.py` | NEW | Entity-tracking → multi-turn text |
| `src/source_monitor/llm/corruption.py` | NEW | Text-level ghost/misloc/phantom |
| `src/source_monitor/llm/provenance.py` | NEW | Token-level self/external tracking |
| `src/source_monitor/llm/telemetry.py` | NEW | Retrospective + emission-time surprisal |
| `src/source_monitor/llm/phase0.py` | NEW | Phase 0 experiment runner |
| `tests/test_task_render.py` | NEW | Task rendering tests |
| `tests/test_corruption_text.py` | NEW | Text corruption tests |
| `tests/test_provenance_llm.py` | NEW | Provenance tracking tests |
| `tests/test_telemetry_llm.py` | NEW | Telemetry / surprisal tests |
| `pyproject.toml` | MODIFY | Add `llm` optional deps |

---

## Verification Plan

### Automated Tests
```bash
# Unit tests (no GPU, no model download — uses fakes)
uv run pytest tests/test_task_render.py tests/test_corruption_text.py tests/test_provenance_llm.py tests/test_telemetry_llm.py -v

# Existing tests still pass
uv run pytest tests/test_model.py tests/test_task.py tests/test_surprise.py -v
```

### Phase 0 Execution
```bash
# Requires Qwen3-1.7B cached locally (one-time download)
# huggingface-cli download Qwen/Qwen3-1.7B
uv sync --extra llm
uv run python -m source_monitor.llm.phase0
```

Expected output: per-corruption-type AUROC + pooled AUROC, printed and saved to `results/llm_phase0_results.jsonl`.

### Success/Failure Decision
- **AUROC ≥ .9 per type AND matched-surface AUROC ≥ .85 (Amendment A2)**:
  Phase 0 PASSES → proceed to Phase 1 (out-of-domain transfer)
- **Pooled ≥ .9 but matched-surface < .85**: the detector is reading surface
  form, not state contradiction — a Phase-0-scale reproduction of the R2
  signature-learning failure. Investigate before proceeding; do NOT report
  the pooled number as a pass.
- **AUROC < .9 but > .7**: Investigate — calibration issue? Span aggregation choice? Task too simple?
- **AUROC ≈ .5**: The signal does not port. Stop. Understand why before any training.

---

## Execution Order

1. Modify `pyproject.toml` (dependencies)
2. `config.py` (no deps beyond stdlib)
3. `task_render.py` + `test_task_render.py` (no deps beyond source_monitor.task)
4. `corruption.py` + `test_corruption_text.py` (depends on task_render)
5. `cache.py` (depends on transformers — imported lazily)
6. `provenance.py` + `test_provenance_llm.py` (depends on tokenizer interface)
7. `telemetry.py` + `test_telemetry_llm.py` (depends on provenance + model)
8. `phase0.py` (orchestrator — depends on all above)
9. `__init__.py` (last — exports)
10. Run tests, then run experiment


---

# Fable Review — Amendments (2026-07-17)

The plan's architecture, test strategy, and decision gates are right and are
approved as written. The amendments below add guards against three
measurement confounds that the toy project's history says to treat as
first-class (every headline there flipped at least once on exactly this
kind of issue), plus two implementation pitfalls specific to chat-templated
LLMs. Opus should fold these in during the numbered execution order —
none change the file structure.

## A1 — Chat-template hygiene (telemetry.py, provenance.py)

Two silent-bug sources specific to Qwen3:

1. **Thinking mode.** Qwen3's chat template inserts `<think>` scaffolding;
   `apply_chat_template(..., enable_thinking=False)` must be pinned for
   every render, and any template-inserted tokens (role markers, think
   tags, BOS/EOS) must be EXCLUDED from the scored span. Score only the
   CONTENT tokens of the assistant turn. Template boilerplate has
   near-deterministic logp and dilutes mean-aggregated surprisal toward
   the ceiling — the toy's equivalent would have been averaging the
   emission's γ with five EMIT markers.
2. **Prefix stability.** The plan's span attribution should be built by
   incremental rendering (template over `turns[:i]`, diff token lengths).
   Add an explicit test: `rendered(turns[:i])` is a string prefix of
   `rendered(turns[:i+1])` for all i (with `add_generation_prompt=False`).
   If the template violates prefix stability anywhere, span indices are
   silently garbage — this is the single most common bug in
   span-attribution code, and the round-trip test as specified would not
   catch a systematic off-by-template-token shift.

## A2 — The surface-form confound (phase0.py analysis; REQUIRED for pass)

In the toy, corrupt and genuine emissions were format-identical single
tokens. In text, "…is in box C" (ghost/misloc) and "…is nowhere" (phantom)
differ in surface statistics from each other. A detector could hit high
pooled AUROC by learning "'nowhere' is a rare continuation," which is a
surface artifact, not state-contradiction reading — the zero-shot analog
of R2's signature-learning failure.

Guard: **matched-surface stratification.** Compare corrupted spans only
against genuine spans of the same claim surface type (container-claims vs
container-claims; nowhere-claims vs nowhere-claims — the clean traces
contain both types naturally, since ~half the emissions are NOWHERE).
Report per-type AUROC both pooled and matched-surface. The pass bar in
the Success/Failure section now requires matched-surface ≥ .85. This is
cheap (a groupby in the analysis) and is the difference between claiming
"the model detects its own false statements" and "the model finds
'nowhere' surprising."

## A3 — Paired scoring and the cascade population (phase0.py)

The plan already generates clean + 3 corrupted variants per trace; use the
pairing:

1. **Primary statistic alongside AUROC:** the paired delta
   `score(span k, corrupted variant) − score(span k, clean variant)` —
   same trace, same position, same surface form up to the planted change.
   This cancels position and content effects that pooled AUROC smears over
   (early spans have less context and higher baseline surprisal).
2. **Three genuine populations, reported separately:** genuine spans in
   clean traces; genuine spans BEFORE the corruption in corrupted traces
   (should match clean); and genuine spans AFTER the corruption
   (contaminated context — the F13b cascade population: TRUE statements
   that contradict the earlier lie may legitimately score as surprising).
   Pooling the cascade population into "genuine" would understate the
   detector; splitting it out turns a nuisance into a diagnostic — an
   elevated post-corruption population is itself evidence the model is
   tracking state rather than surface.

## A4 — Aggregation as a reported ablation, not a config choice

Aggregation is where the signal lives or dies: most tokens of "The red
ball is in box A" are boilerplate; the information is in the location
tokens. Report three aggregations side by side rather than defaulting to
one: (a) mean −logp over content tokens, (b) max token −logp (= min logp)
over content tokens, (c) location-slot only — the renderer knows exactly
which tokens carry the claim; scoring only those is the toy-faithful
version and the expected ceiling. If (c) ≫ (a), that quantifies how much
Phase 1's free-text setting (where slots are unknown) will need from
smarter span segmentation. This reframes `config.aggregation` from a
choice into three columns of the results table.

## A5 — Minor

- `n_ops` at 8 keeps ~200-token contexts; the harder 12×6×4 config (Open
  Question 1 resolution) roughly doubles that. Both are trivial for the
  3060.
- Log `transformers`, model revision hashes, and template flags into the
  JSONL records (basin-variance discipline: every number traceable to an
  exact environment).
- `emission_time_surprisal` stays implemented-but-unused in Phase 0, as
  planned; its generation path must also pin `enable_thinking=False` when
  it is eventually exercised.
- Fan protocol applies from the first model load.

## Verdict

Proceed in the plan's execution order with A1–A4 folded in. The plan's
instinct to fail closed (stop at AUROC ≈ .5 and understand) is exactly
right; A2 extends the same instinct to the more dangerous failure mode —
passing for the wrong reason.


---

# Phase 0b — Contrastive Slot Scoring (Fable, post-Phase-0 review)

**Trigger.** Phase 0's decision gate landed in the INVESTIGATE branch on
phantom: matched-surface AUROC .778-.838 (< .85) in every configuration,
while both container types passed at .97-1.00. The ghost POOLED anomaly
(.835 pooled vs .990 matched, slot aggregation) shares the cause:
**absence-assertion prior bias** — the model assigns depressed logp to
"nowhere"-claims regardless of truth, so true absences look surprising and
false ones cannot separate from them. Raw surprisal conflates
"improbable claim-form" with "contradicts state." See FINDINGS F16.

## The fix

Replace raw span surprisal with a **candidate-renormalized (likelihood-
ratio) score** at each claim slot. For emission slot s with candidate set
C(s) = {each container named in the trace} ∪ {"nowhere"}:

1. Render the assistant turn once per candidate c ∈ C(s) (same prefix,
   only the location phrase substituted).
2. Teacher-forced score of the slot tokens for each candidate:
   lp(c) = sum logp(slot tokens of c | prefix).  (Length-normalize:
   mean logp per slot token, so "box A" vs "nowhere" token counts don't
   bias the comparison.)
3. Signal for the emitted claim e:
   score(s) = −[ lp(e) − logsumexp_{c∈C(s)} lp(c) ]
   i.e. −log of the renormalized probability of the emitted claim within
   the candidate set. Class priors common to a surface form cancel; what
   remains is the model's contextual preference ORDERING — the
   re-derive-and-compare (cavity) check restricted to the answer space.

## Implementation notes (one function + one analysis pass)

- `telemetry.py`: add `contrastive_slot_score(model, trace_tokens,
  slot_annotations, candidates)` — batches the |C| variants of each turn;
  prefix is shared so per-slot cost is |C| short suffix scorings. At 400
  traces × ≤12 slots × ≤5 candidates this is minutes on the 3060.
- `phase0.py`: add `--scoring contrastive` alongside `raw`; emit the same
  pooled / matched / paired tables, PLUS the A3 three-population split
  (clean-trace genuine, pre-corruption genuine, post-corruption genuine)
  which Phase 0 left unreported — the post-corruption population is the
  cascade diagnostic (F13b) and should be examined, not pooled away.
- Keep Phase 0's raw-scoring results as the comparison column; the delta
  between raw and contrastive per type IS the measured size of the prior
  bias — report it.

## Pre-registered predictions and gate

- P-0b.1: phantom matched-surface AUROC ≥ .90 under contrastive scoring
  (1.7B, both task configs).
- P-0b.2: ghost pooled ≈ ghost matched (the anomaly dissolves) — the
  internal-consistency check that the mechanism story is right.
- P-0b.3: container types unchanged (≥ .97 matched).
- PASS: all three → Phase 0 closes as PASS-with-mechanism-note; proceed
  to Phase 1 carrying BOTH scorers (raw for free-text where candidates
  can't be enumerated; contrastive wherever a claim's alternative set is
  constructible — and note that Phase 1 should expect raw surprisal to
  underperform on negative/absence assertions generally, which is now a
  predicted OOD failure mode, not a surprise).
- FAIL on P-0b.1 with priors cancelled: the model genuinely cannot track
  absence at these scales — a real capability limit; characterize (does
  it scale 0.6B→1.7B→4B?) before any Phase 2 training.
