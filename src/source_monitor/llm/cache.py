"""Fail-closed model loading for LLM experiments.

Loads Qwen3 (or compatible) checkpoints from the local Hugging Face cache.
Never retries with network access — if the snapshot isn't cached, errors
with an explicit download command.  Adapted from seer's cache.py, stripped
to essentials for inference-only use.

A5: logs transformers version, model revision, and template flags into
returned metadata for JSONL provenance records.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ModelLoadError(RuntimeError):
    """The requested model cannot be safely loaded from the local cache."""


@dataclass(frozen=True)
class ModelMeta:
    """Provenance metadata recorded alongside every experiment result (A5)."""
    model_name: str
    revision: str | None
    transformers_version: str
    torch_version: str
    python_version: str
    device: str
    dtype: str
    enable_thinking: bool

    def to_dict(self) -> dict[str, Any]:
        """Flat dict suitable for JSONL serialization."""
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "device": self.device,
            "dtype": self.dtype,
            "enable_thinking": self.enable_thinking,
        }


MIN_TRANSFORMERS_VERSION = (4, 51)


def _require_transformers_version() -> str:
    """Verify transformers is installed and recent enough. Returns version string."""
    try:
        import transformers
    except ImportError as error:
        raise ModelLoadError(
            "transformers >=4.51 is required; install with: "
            "uv sync --extra llm"
        ) from error
    version = transformers.__version__
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < MIN_TRANSFORMERS_VERSION:
        raise ModelLoadError(
            f"transformers >=4.51 required; found {version!r}. "
            "Update with: uv sync --extra llm"
        )
    return version


def format_download_command(model_name: str) -> str:
    """Return an explicit opt-in download command; this module never runs it."""
    return " ".join(
        shlex.quote(part) for part in ["huggingface-cli", "download", model_name]
    )


_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16",
    "float16": "torch.float16",
    "float32": "torch.float32",
}


def load_model(
    model_name: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    enable_thinking: bool = False,
) -> tuple[Any, Any, ModelMeta]:
    """Load a pretrained causal LM + tokenizer from local HF cache.

    Returns (model, tokenizer, metadata).

    The model is placed in eval mode with no_grad context expected from caller.
    Never downloads — errors with instructions if the model isn't cached.

    Parameters
    ----------
    model_name:
        Hugging Face model ID, e.g. "Qwen/Qwen3-1.7B".
    device:
        Target device, e.g. "cuda" or "cpu".
    dtype:
        Weight precision: "bfloat16", "float16", or "float32".
    enable_thinking:
        A1: must be False for Phase 0. Pinned here and propagated to
        tokenizer.apply_chat_template calls.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    transformers_version = _require_transformers_version()

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ModelLoadError(f"Unknown dtype {dtype!r}; expected one of {list(_DTYPE_MAP)}")

    CHATML_TEMPLATE = (
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|im_start|>assistant\\n' -}}"
        "{%- endif -%}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.chat_template = CHATML_TEMPLATE
    except Exception as error:
        raise ModelLoadError(
            f"Cannot load tokenizer for {model_name!r} from local cache. "
            f"Download first: {format_download_command(model_name)}"
        ) from error

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
        )
    except Exception as error:
        raise ModelLoadError(
            f"Cannot load model {model_name!r} from local cache. "
            f"Download first: {format_download_command(model_name)}"
        ) from error

    model = model.to(device)
    model.eval()

    # Resolve revision from the model config if available
    revision: str | None = None
    if hasattr(model.config, "_commit_hash"):
        revision = model.config._commit_hash

    meta = ModelMeta(
        model_name=model_name,
        revision=revision,
        transformers_version=transformers_version,
        torch_version=torch.__version__,
        python_version=sys.version,
        device=device,
        dtype=dtype,
        enable_thinking=enable_thinking,
    )

    return model, tokenizer, meta
