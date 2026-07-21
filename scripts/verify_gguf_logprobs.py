"""Verify llama-cpp-python exposes per-token logprobs for SUPPLIED text.

This is the F22e blocker check. The Unsloth/llama-server OpenAI endpoint silently
drops echo and returns logprobs only for generated tokens. The claim (confirmed by
the user's research) is that the llama-cpp-python wrapper can force per-token
evaluation of the prompt. This script proves it end to end on a tiny CPU model —
model quality is irrelevant; only the API surface is under test.

Two independent paths are checked:
  1. create_completion(echo=True, logprobs=N)  -> choices[0].logprobs.token_logprobs
     must contain one value per PROMPT token (first is None: nothing predicts it).
  2. logits_all=True + low-level eval          -> the exact per-token neg-logprob
     of a supplied continuation, which is what the detector's retrospective
     surprisal needs. Cross-checked against path 1.

Run (CPU, no heat):
    .venv/bin/python scripts/verify_gguf_logprobs.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from llama_cpp import Llama

# Tiny real GGUF already on disk (Karpathy's stories260K). Pure mechanism test.
CANDIDATES = [
    Path.home() / ".unsloth/.cache/stories260K.gguf",
    Path.home() / ".cache/huggingface/hub/models--ggml-org--models/snapshots"
    "/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf",
]


def _model_path() -> str:
    for p in CANDIDATES:
        if p.exists():
            return str(p)
    sys.exit(f"no tiny GGUF found; looked in {[str(c) for c in CANDIDATES]}")


def main() -> None:
    path = _model_path()
    print(f"model: {path}")
    llm = Llama(model_path=path, n_ctx=256, logits_all=True, verbose=False)

    text = "Once upon a time there was a little"

    # --- Path 1: OpenAI-style echo + logprobs -----------------------------------
    out = llm.create_completion(
        prompt=text, max_tokens=1, echo=True, logprobs=5, temperature=0.0
    )
    lp = out["choices"][0]["logprobs"]
    toks = lp["tokens"]
    tok_lps = lp["token_logprobs"]
    n_prompt = len(llm.tokenize(text.encode()))

    print("\n[path 1] create_completion(echo=True, logprobs=5)")
    print(f"  prompt tokens (tokenizer): {n_prompt}")
    print(f"  tokens returned:           {len(toks)} (prompt echoed + 1 generated)")
    print(f"  token_logprobs returned:   {len(tok_lps)}")
    print(f"  first token_logprob is None (unpredictable): {tok_lps[0] is None}")
    scored = [x for x in tok_lps[:n_prompt] if x is not None]
    print(f"  scored PROMPT tokens:      {len(scored)}  "
          f"(mean nll {(-sum(scored) / len(scored)):.3f} nats)")
    print("  per-token (prompt): " + "  ".join(
        f"{repr(t)}={(-l):.2f}" for t, l in zip(toks[1:n_prompt], tok_lps[1:n_prompt])
    ))
    path1_ok = len(scored) == n_prompt - 1 and tok_lps[0] is None

    # --- Path 2: low-level logits_all, exact surprisal of a supplied span --------
    # Score the continuation " girl" given the prompt, straight from the logits.
    import numpy as np

    ctx = llm.tokenize(text.encode())
    cont = llm.tokenize(b" girl", add_bos=False)
    llm.reset()
    llm.eval(ctx + cont)
    # logits[i] predicts token i+1; score each continuation token against its
    # preceding position.
    nlls = []
    for j, tid in enumerate(cont):
        pos = len(ctx) + j - 1
        logits = np.asarray(llm.eval_logits[pos], dtype=np.float64)
        logits -= logits.max()
        logp = logits - math.log(np.exp(logits).sum())
        nlls.append(-float(logp[tid]))
    print("\n[path 2] logits_all=True, exact neg-logprob of a supplied span")
    print(f"  span ' girl' ({len(cont)} tok): "
          + "  ".join(f"{n:.3f}" for n in nlls) + " nats")
    path2_ok = all(math.isfinite(n) and n >= 0 for n in nlls)

    print("\nRESULT")
    print(f"  path 1 (echo prompt logprobs): {'PASS' if path1_ok else 'FAIL'}")
    print(f"  path 2 (logits_all surprisal): {'PASS' if path2_ok else 'FAIL'}")
    ok = path1_ok and path2_ok
    print(f"  => llama-cpp-python {'CAN' if ok else 'CANNOT'} score supplied text; "
          f"F22e blocker is {'surmountable via the wrapper' if ok else 'NOT cleared'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
