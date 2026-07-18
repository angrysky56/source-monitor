"""
Ghost-mix training for the source-monitored decoder.

Training traces are a mixture: with probability --ghost-frac, a ghost-able
trace (query object removed) is presented WITH its corrupted self-emission,
but the loss targets are ALWAYS the true locations — the model is trained to
emit the truth even when its own prior emission in the context lies. This is
the pressure that makes an admission mechanism worth learning at all; a
clean-trained model has never seen a lying self-emission (that arm replicates
the original blind spot).

Loss = CE at EMIT markers (true targets)
     + gate_loss_w * BCE on gate logits at self positions   [gate="sup" only]
       (label 1 = genuine emission, 0 = the corrupted one)

The CRITICAL control: the plain baseline must also be trained on the same
ghost mixture (arm `base-mix`), otherwise mechanism and data augmentation are
confounded. See SPEC.md §5.

CLI:
    python -m source_monitor.train --provenance 1 --gate sup --ghost-frac 0.3
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import SMConfig, SMDecoder
from .muon import build_optimizers
from .task import (
    VOCAB_SIZE,
    Task,
    generate_dataset,
    inject_ghost,
    pad_batch,
    provenance_ids,
    self_positions,
)


def make_tensor(tasks: list[Task], device: torch.device) -> torch.Tensor:
    padded, _ = pad_batch(tasks)
    return torch.tensor(padded, dtype=torch.long, device=device)


def prov_tensor(tasks: list[Task], device: torch.device) -> torch.Tensor:
    L = max(len(t.tokens) for t in tasks)
    return torch.tensor([provenance_ids(t, L) for t in tasks],
                        dtype=torch.long, device=device)


def loc_target_tensor(tasks: list[Task], device: torch.device) -> torch.Tensor:
    return torch.tensor([t.loc_targets for t in tasks], dtype=torch.long, device=device)


@dataclass
class Bundle:
    """Pre-tensorized training data: clean and ghosted variants of each trace."""
    x_clean: torch.Tensor    # (N, L)
    x_ghost: torch.Tensor    # (N, L) == x_clean where no ghost is injectable
    has_ghost: torch.Tensor  # (N,) bool
    ghost_step: torch.Tensor # (N,) long, -1 where none
    targets: torch.Tensor    # (N, n_emit) TRUE loc targets (never corrupted)
    prov: torch.Tensor       # (N, L) provenance ids (identical rows; fixed layout)
    emit_pos: list[int]      # EMIT marker indices (loss/readout positions)
    self_pos: list[int]      # self-emitted loc indices (gate positions)


def build_bundle(tasks: list[Task], device: torch.device, seed: int) -> Bundle:
    assert len({len(t.tokens) for t in tasks}) == 1, "fixed-length tasks required"
    rng = random.Random(seed)
    ghosts: list[Task] = []
    has: list[bool] = []
    steps: list[int] = []
    for t in tasks:
        g = inject_ghost(t, rng)
        if g is None:
            ghosts.append(t); has.append(False); steps.append(-1)
        else:
            k = next(i for i in range(len(t.loc_targets))
                     if t.loc_targets[i] != g.loc_targets[i])
            ghosts.append(g); has.append(True); steps.append(k)
    return Bundle(
        x_clean=make_tensor(tasks, device),
        x_ghost=make_tensor(ghosts, device),
        has_ghost=torch.tensor(has, dtype=torch.bool, device=device),
        ghost_step=torch.tensor(steps, dtype=torch.long, device=device),
        targets=loc_target_tensor(tasks, device),
        prov=prov_tensor(tasks, device),
        emit_pos=list(tasks[0].emit_marker_pos),
        self_pos=self_positions(tasks[0]),
    )


def batch_loss(
    model: SMDecoder, bundle: Bundle, idx: torch.Tensor,
    ghost_frac: float, gate_loss_w: float, generator: torch.Generator | None = None,
    emission_dropout: float = 0.0,
) -> torch.Tensor:
    """Mixture loss for one sampled batch of row indices."""
    b = idx.shape[0]
    dev = idx.device
    coin = torch.rand(b, device=dev, generator=generator) < ghost_frac
    use_g = coin & bundle.has_ghost[idx]
    x = torch.where(use_g[:, None], bundle.x_ghost[idx], bundle.x_clean[idx])
    prov = bundle.prov[idx]
    drop = None
    if emission_dropout > 0.0:
        # v3: hard-mask a random subset of self-emissions — rehearse the hole
        drop = (torch.rand(x.shape, device=dev, generator=generator)
                < emission_dropout) & prov.bool()
    logits, gate_logit = model(x, prov, drop=drop)
    pred = logits[:, bundle.emit_pos, :]
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                           bundle.targets[idx].reshape(-1))
    if model.cfg.gate == "sup" and gate_logit is not None:
        labels = torch.ones(b, len(bundle.self_pos), device=dev)
        rows = torch.nonzero(use_g, as_tuple=True)[0]
        labels[rows, bundle.ghost_step[idx][rows]] = 0.0
        gl = gate_logit[:, bundle.self_pos]
        loss = loss + gate_loss_w * F.binary_cross_entropy_with_logits(gl, labels)
    return loss


@torch.no_grad()
def evaluate(model: SMDecoder, tasks: list[Task], device: torch.device,
             batch: int = 512) -> dict[str, float]:
    """Clean-trace competence: loss + final-answer accuracy + per-step accuracy."""
    model.eval()
    emit_pos = list(tasks[0].emit_marker_pos)
    losses: list[float] = []
    correct = total = 0
    abs_c = abs_t = pres_c = pres_t = 0
    step_correct = step_total = 0
    for i in range(0, len(tasks), batch):
        chunk = tasks[i : i + batch]
        x = make_tensor(chunk, device)
        pv = prov_tensor(chunk, device)
        tgt = loc_target_tensor(chunk, device)
        logits, _ = model(x, pv)
        pred = logits[:, emit_pos, :]
        losses.append(F.cross_entropy(pred.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)).item())
        preds = pred.argmax(-1)
        step_correct += int((preds == tgt).sum())
        step_total += tgt.numel()
        final = preds[:, -1]
        for j, t in enumerate(chunk):
            ok = int(int(final[j]) == t.answer)
            correct += ok; total += 1
            if t.final_absent:
                abs_c += ok; abs_t += 1
            else:
                pres_c += ok; pres_t += 1
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "acc": correct / total,
        "acc_absent": abs_c / abs_t if abs_t else float("nan"),
        "acc_present": pres_c / pres_t if pres_t else float("nan"),
        "acc_perstep": step_correct / step_total,
    }


def train_inline(cfg: SMConfig, args: argparse.Namespace,
                 device: torch.device) -> SMDecoder:
    """Train one model per `args` (shared by the CLI and experiments.py)."""
    torch.manual_seed(args.seed)
    tkw = dict(n_ops=args.n_ops, n_objects=args.n_objects, n_containers=args.n_containers)
    tasks = generate_dataset(args.train_tasks, base_seed=1, **tkw)
    bundle = build_bundle(tasks, device, seed=args.seed + 10_000)
    model = SMDecoder(cfg).to(device)
    opts = build_optimizers(model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr,
                            use_muon=bool(args.use_muon))
    n = bundle.x_clean.shape[0]
    use_amp = device.type == "cuda"
    e_drop = getattr(args, "emission_dropout", 0.0)
    for _step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = batch_loss(model, bundle, idx, args.ghost_frac, args.gate_loss_w,
                              emission_dropout=e_drop)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
    return model


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--use-muon", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--gate-layer", type=int, default=3)
    ap.add_argument("--n-ops", type=int, default=8)
    ap.add_argument("--n-objects", type=int, default=4)
    ap.add_argument("--n-containers", type=int, default=3)
    ap.add_argument("--adamw-lr", type=float, default=3e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--train-tasks", type=int, default=20000)
    ap.add_argument("--val-tasks", type=int, default=2000)
    ap.add_argument("--ghost-frac", type=float, default=0.3)
    ap.add_argument("--gate-loss-w", type=float, default=0.5)
    ap.add_argument("--emission-dropout", type=float, default=0.0,
                    help="v3: train-time hard-mask rate on self-emissions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)


def make_config(args: argparse.Namespace, provenance: bool, gate: str,
                gate_hard: bool = False) -> SMConfig:
    return SMConfig(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=4 * args.d_model,
        provenance=provenance, gate=gate, gate_layer=args.gate_layer,
        gate_hard=gate_hard,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--provenance", type=int, default=1)
    ap.add_argument("--gate", choices=["none", "task", "sup", "surprise"], default="sup")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        print("NOTE: GPU run — 3060 fan is broken; start the floor fans and "
              "watch nvidia-smi.", flush=True)
    torch.manual_seed(args.seed)
    tkw = dict(n_ops=args.n_ops, n_objects=args.n_objects, n_containers=args.n_containers)
    tasks = generate_dataset(args.train_tasks, base_seed=1, **tkw)
    val = generate_dataset(args.val_tasks, base_seed=999, **tkw)
    bundle = build_bundle(tasks, dev, seed=args.seed + 10_000)
    cfg = make_config(args, provenance=bool(args.provenance), gate=args.gate)
    model = SMDecoder(cfg).to(dev)
    opts = build_optimizers(model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr,
                            use_muon=bool(args.use_muon))
    tag = f"prov{int(cfg.provenance)}/gate-{cfg.gate}/gf{args.ghost_frac}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params/1e6:.2f}M  seq_len={bundle.x_clean.shape[1]}  "
          f"ghostable={int(bundle.has_ghost.sum())}/{len(tasks)}", flush=True)

    n = bundle.x_clean.shape[0]
    use_amp = dev.type == "cuda"
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = batch_loss(model, bundle, idx, args.ghost_frac, args.gate_loss_w,
                              emission_dropout=args.emission_dropout)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, val, dev)
            print(f"[{tag}] step {step:5d}  loss {m['loss']:.4f}  "
                  f"answer_acc {m['acc']:.3f} (absent {m['acc_absent']:.3f} / "
                  f"present {m['acc_present']:.3f})  perstep {m['acc_perstep']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if args.out:
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, args.out)
        print(f"[{tag}] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
