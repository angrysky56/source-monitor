"""Phase 2: hole-rehearsal LoRA (the repair leg at LLM scale).

Trains Qwen3 with a fraction of its own prior emissions attention-masked out
(hard holes), forcing re-derivation from external evidence — the LLM analog of
the toy's emission dropout (R5 / F13b-F14). See implementation_plan_phase2.md.
"""
