"""Phase 2 hole-rehearsal LoRA training loop (bf16, Qwen3-1.7B).

Trains three arms (base / drop / corrupt), saves LoRA adapters, and evaluates each
against the un-fine-tuned base with eval_repair. Minimal custom loop (no trl) so
the hole-masking collator controls the attention mask directly.

Requires `peft` (imported lazily). bf16 LoRA on 1.7B fits the 12 GB 3060; use the
fan protocol. Run: python -m source_monitor.llm.train.lora_train
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from source_monitor.llm.cache import load_model
from source_monitor.llm.train.config import Phase2Config
from source_monitor.llm.train.data import HoleCollator, build_examples, iterate_batches
from source_monitor.llm.train.eval_repair import evaluate, load_for_eval


def _write(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def train_arm(cfg: Phase2Config, arm: str, seed: int, adapter_root: Path) -> str:
    from peft import LoraConfig, get_peft_model

    model, tok, _meta = load_model(cfg.model_name, device=cfg.device, dtype=cfg.dtype,
                                   enable_thinking=False)
    lc = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lc)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    mode = "corrupt" if arm == "corrupt" else "clean"
    p_hole = cfg.p_hole if arm == "drop" else 0.0
    exs = build_examples(tok, seed, cfg.n_train, cfg.domains, mode=mode)
    collate = HoleCollator(p_hole, getattr(tok, "pad_token_id", 0), seed=seed)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    for step, batch in enumerate(iterate_batches(exs, cfg.batch_size, cfg.steps, seed)):
        b = {k: v.to(cfg.device) for k, v in collate(batch).items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        shift_labels = b["labels"][:, 1:]
        mask = shift_labels != -100
        # CE only on supervised positions: avoids materializing a full-vocab
        # float32 tensor over ALL positions (the OOM source on the 12GB card).
        sel_logits = out.logits[:, :-1, :][mask]  # (n_supervised, vocab)
        loss = F.cross_entropy(sel_logits.float(), shift_labels[mask])
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(f"[{arm} seed{seed}] step {step}/{cfg.steps} loss {loss.item():.4f}", flush=True)

    out_dir = adapter_root / f"{arm}_seed{seed}"
    model.save_pretrained(str(out_dir))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 hole-rehearsal LoRA")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    args = ap.parse_args()

    cfg = Phase2Config()
    cfg = replace(
        cfg,
        arms=tuple(args.arms) if args.arms else cfg.arms,
        seeds=tuple(args.seeds) if args.seeds else cfg.seeds,
        steps=args.steps or cfg.steps,
        n_train=args.n_train or cfg.n_train,
        n_eval=args.n_eval or cfg.n_eval,
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    adapter_root = Path(cfg.results_dir) / "phase2_adapters"
    out_file = Path(cfg.results_dir) / "llm_phase2_results.jsonl"

    # Resume: skip (arm, seed) already persisted.
    done: set[tuple[str, int]] = set()
    if out_file.exists():
        for line in open(out_file, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["arm"], int(r["seed"])))

    # Baseline: un-fine-tuned model, once per seed (skip completed).
    base_todo = [s for s in cfg.seeds if ("base_noft", s) not in done]
    if base_todo:
        base, tok, _ = load_model(cfg.model_name, device=cfg.device, dtype=cfg.dtype,
                                  enable_thinking=False)
        base.eval()
        for seed in base_todo:
            m = evaluate(base, tok, seed, cfg.n_eval, cfg.device)
            _write(out_file, {"arm": "base_noft", "seed": seed, "model_name": cfg.model_name, **m})
            print("base_noft", seed, m, flush=True)
        del base
        gc.collect()
        torch.cuda.empty_cache()

    for arm in cfg.arms:
        for seed in cfg.seeds:
            if (arm, seed) in done:
                print(f"skip {arm} seed{seed} (already done)", flush=True)
                continue
            t0 = time.time()
            adir = train_arm(cfg, arm, seed, adapter_root)
            model, tok, _ = load_for_eval(cfg.model_name, cfg.device, cfg.dtype, adir)
            m = evaluate(model, tok, seed, cfg.n_eval, cfg.device)
            _write(out_file, {"arm": arm, "seed": seed, "wall_s": round(time.time() - t0, 1),
                              "model_name": cfg.model_name, "adapter": adir, **m})
            print(arm, seed, m, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
