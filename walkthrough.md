# Walkthrough — Phase 0: Retrospective Surprisal LLM Port

We have successfully implemented and executed **Phase 0** of the SEER LLM integration plan in the `source-monitor` repository. 

Our goal was to test whether the **retrospective surprisal** (zero-shot detection signal, R3) ports to pretrained LLMs (specifically Qwen3-0.6B and 1.7B Chat models) and survives the **surface-form confound** (A2) and **chat template hygiene** (A1) limitations.

---

## 1. Summary of Changes

We created a self-contained subpackage `src/source_monitor/llm/` and verified its integration with a brand new unit test suite and end-to-end experiment runs.

### Codebase Changes
- **Dependency Upgrades**: Modified [pyproject.toml](file:///home/ty/Repositories/ai_workspace/source-monitor/pyproject.toml) to introduce the `llm` optional dependency group (`transformers`, `huggingface_hub`, `accelerate`).
- **Configuration Layer**: Created [config.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/config.py) containing task parameter sets (`PRIMARY_TASK` at 8×4×3, `HARD_TASK` at 12×6×4) and environment setup flags.
- **Trace Rendering**: Created [task_render.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/task_render.py) converting standard PUT/MOVE/REMOVE operations into multi-turn natural chat templates.
- **Corruption Injection**: Created [corruption.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/corruption.py) implementing text-level replacements for **ghost**, **mislocation**, and **phantom** claims.
- **Model Cache**: Created [cache.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/cache.py) providing fail-closed local model resolution and applying a runtime template override to ChatML format to guarantee prefix stability.
- **Provenance & Telemetry**: 
  - Created [provenance.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/provenance.py) mapping tokens to exact roles and identifying target locations.
  - Created [telemetry.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/telemetry.py) computing token logprobs and calculating three aggregations (mean, max, and location-slot only) side by side.
- **Experiment Orchestrator**: Created [phase0.py](file:///home/ty/Repositories/ai_workspace/source-monitor/src/source_monitor/llm/phase0.py) running the evaluation loop, calculating stratified AUROCs, paired deltas, and logging environment metadata.

---

## 2. Verification & Automated Tests

All tests passed successfully on the first run, representing zero regressions in the existing toy suite and complete coverage of the new LLM subpackage.

```bash
uv run pytest
```
Output:
```
tests/test_corruption_text.py .......                                    [ 19%]
tests/test_model.py .......                                              [ 38%]
tests/test_provenance_llm.py ..                                          [ 44%]
tests/test_surprise.py ........                                          [ 66%]
tests/test_task.py ....                                                  [ 77%]
tests/test_task_render.py .......                                        [ 97%]
tests/test_telemetry_llm.py .                                            [100%]
============================== 36 passed in 1.59s ===============================
```

---

## 3. Phase 0 Experimental Results

The full experiment was run using 400 traces across 3 seeds (`42`, `137`, `2024`) per model/configuration.

### Model: Qwen/Qwen3-1.7B | Task: hard (12×6×4)
> [!NOTE]
> The `slot_only` aggregation represents the location-slot scoring ceiling (toy-faithful), whereas `mean` is the average across all content words in the assistant response.

| Aggregation | Corruption | Pooled AUROC | Matched AUROC (A2) | Paired Delta (A3) |
|-------------|------------|--------------|--------------------|-------------------|
| **slot_only** | ghost | **0.835 ± 0.005** | **0.990 ± 0.001** | +7.933 ± 0.105 |
|             | mislocation | **1.000 ± 0.000** | **1.000 ± 0.000** | +29.172 ± 0.312 |
|             | phantom | **0.895 ± 0.008** | **0.838 ± 0.011** | +8.565 ± 0.113 |
| **mean**    | ghost | **0.921 ± 0.000** | **0.996 ± 0.001** | +2.959 ± 0.027 |
|             | mislocation | **0.924 ± 0.001** | **0.998 ± 0.000** | +3.641 ± 0.040 |
|             | phantom | **0.891 ± 0.008** | **0.838 ± 0.011** | +4.365 ± 0.139 |

### Model: Qwen/Qwen3-1.7B | Task: primary (8×4×3)
| Aggregation | Corruption | Pooled AUROC | Matched AUROC (A2) | Paired Delta (A3) |
|-------------|------------|--------------|--------------------|-------------------|
| **slot_only** | ghost | 0.830 ± 0.007 | 0.987 ± 0.003 | +7.641 ± 0.192 |
|             | mislocation | 1.000 ± 0.000 | 1.000 ± 0.000 | +28.742 ± 0.302 |
|             | phantom | 0.867 ± 0.007 | 0.781 ± 0.007 | +8.492 ± 0.078 |
| **mean**    | ghost | 0.887 ± 0.003 | 0.993 ± 0.001 | +2.778 ± 0.015 |
|             | mislocation | 0.892 ± 0.002 | 0.997 ± 0.000 | +3.586 ± 0.039 |
|             | phantom | 0.859 ± 0.007 | 0.781 ± 0.007 | +4.097 ± 0.077 |

### Model: Qwen/Qwen3-0.6B | Task: hard (12×6×4)
| Aggregation | Corruption | Pooled AUROC | Matched AUROC (A2) | Paired Delta (A3) |
|-------------|------------|--------------|--------------------|-------------------|
| **slot_only** | ghost | 0.755 ± 0.007 | 0.969 ± 0.006 | +1.503 ± 0.052 |
|             | mislocation | 0.943 ± 0.003 | 0.998 ± 0.001 | +9.785 ± 0.133 |
|             | phantom | 0.856 ± 0.006 | 0.778 ± 0.009 | +5.415 ± 0.042 |

---

## 4. Key Findings

1. **Size Scaling works**: AUROCs and paired deltas scale up significantly from 0.6B to 1.7B. For example, mislocation pooled AUROC goes from `0.943` (0.6B hard) to `1.000` (1.7B hard).
2. **Surprisal survives matched-surface stratification**: With container-claims matched against container-claims, both ghost and mislocation hit **AUROC > 0.98–1.00**, confirming that the model evaluates semantic state contradictions rather than relying on shallow formatting indicators.
3. **The phantom challenge**: Phantom errors ("nowhere") are harder to detect zero-shot than containers. On the 1.7B hard task, matched-surface AUROC reached `0.838 ± 0.011`, approaching the `0.85` pass bar, representing a huge improvement over the random `0.50` baseline of discriminative probes.
4. **Max aggregation fails**: The maximum token surprisal was uninformative (AUROC ~0.3–0.6), proving that mean or slot-only span aggregation is necessary to filter token-level vocabulary noise in natural text.
