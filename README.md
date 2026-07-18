# source-monitor

**Can a model learn not to believe its own echo?**

The self-correction blind spot, reframed as a **source-monitoring failure**:
a standard decoder treats its own emissions as fresh external evidence
(backtracking inference, in belief-propagation terms). This project gives a
small decoder the two things that frame says it is missing:

1. **Provenance** — a learned self/external origin embedding (information the
   system always has and the standard architecture discards), and
2. **An admission gate** — a depth-causal, zero-init scalar head that can
   softly evict self-emitted tokens from attention when they look
   untrustworthy (`logsigmoid(γ)` added to attention logits toward that key).

Built directly on the `sps-blindspot` terminal conclusion (SPS separation is
orthogonal to the blind spot; the fix must gate what enters authoritative
state) and its validated instruments (ghost protocol, JVP amplification,
native confidence head at AUROC 0.955–0.968). Task, Muon, and instruments are
vendored — self-contained, no cross-project imports.

**The project ran to completion 2026-07-17.** Clear narrative report:
**[WRITEUP.md](WRITEUP.md)**. Chronological lab log with every dead end:
**[FINDINGS.md](FINDINGS.md)**. LLM/seer retrofit plan (Qwen3 / Gemma 4):
**[SEER-INTEGRATION.md](SEER-INTEGRATION.md)**. Original design + predictions
P1–P9: **[SPEC.md](SPEC.md)**.

## Layout

- `src/source_monitor/model.py` — decoder + provenance embedding + admission gate
- `src/source_monitor/task.py` — entity tracking w/ dense emission; ghost
  (trained-on) + mislocation (held-out) corruptions; provenance ids
- `src/source_monitor/train.py` — ghost-mix training, joint task+gate loss
- `src/source_monitor/blindspot.py` — behavioral protocol + gate diagnostics
- `src/source_monitor/amplification.py` — JVP Jacobian triplet (selective contraction)
- `src/source_monitor/experiments.py` — all arms × seeds, one process, durable JSONL
- `tests/` — mask causality, gate targeting/liveness, loss paths, JVP-through-gate

## Run

```bash
uv sync                # torch (~2.5GB if not already in the uv cache)
uv run pytest          # guardrails first — CPU, fast

# the five-arm comparison (GPU: ~2 min/arm at defaults on the 3060 —
# fan is broken: floor fans on, watch nvidia-smi)
uv run python -m source_monitor.experiments --seeds 0,1,2

# single arm / manual control
uv run python -m source_monitor.train --provenance 1 --gate sup --ghost-frac 0.3
```

## The one table that matters (predictions, see SPEC §4)

| arm | d1 after ghost | gate closes on ghost? | transfers to held-out corruption? |
|---|---|---|---|
| base-clean | 0.30 ✔ anchor | — | — |
| base-mix | 0.97 (data alone suffices) | — | ✗ (type-bound) |
| gate-sup | 0.99 | AUROC 1.000 ✔ | ✗ AUROC ~.68, never closes |
| gate-task | 0.98 (from data, not gate) | ✗ chance (no emergence) | ✗ |
| surp-clean (v2) | ? | predicted ≥0.9, zero corruption exposure | predicted YES — the live question |

Seeds 0–2 verdicts in FINDINGS.md: supervised admission is
corruption-type-bound (the L1 story at mechanism level). v2 (`gate="surprise"`,
γ = a·logp(emitted | own prediction) + b, label-free) tests whether a
generative self-consistency signal transfers where discriminative supervision
did not — run `--arms surp-clean,surp-mix` with a third held-out corruption
type (phantom removal) now in the matrix.

## Relation to the wider work

`seer`'s Admission pillar, built and instrumented at toy scale. The held-out
corruption test (P4) is the miniature of the domain-transfer question the
token probe failed (L1) and the energy channel must answer. If the gate
transfers, admission-by-internal-signal is learnable; wire the same pattern to
`efh-core`'s external verification for formalizable claims.
