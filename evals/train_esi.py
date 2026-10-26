"""Fine-tune the ESI extras of MambaESI on top of a frozen Mamba backbone.

What this script trains
-----------------------
Only the three ESI-specific modules:

* ``model.question_embedding``       (Linear d_model -> d_model)
* ``model.backbone.embedding_proj``  (Linear d_model -> d_model)
* ``model.backbone.injection_proj``  (Linear d_model -> d_model)

For a 130M backbone that is roughly 1.8M trainable parameters (~1.4% of the
total). The original Mamba backbone (and ``lm_head`` / embeddings) is frozen.

Data
----
By default uses 200 examples from the public SQuAD v1.1 train split as a small
proof-of-life dataset (single A100-minutes worth of compute, or a few minutes
on a free Colab T4). Each example is formatted as::

    Context: <passage>
    Question: <question>
    Answer:<answer>

The model is asked to predict only the ``<answer>`` tokens — context and
question tokens are masked out of the loss (``-100``). The ESI mechanism
receives the raw question string as ``question_ids``, so the search has to
learn to retrieve the answer-relevant tokens out of the context.

Outputs
-------
* ``--esi-out`` (default ``evals/esi_extras.pt``): state-dict of just the ESI
  modules, loadable into any MambaESI of the same config.
* ``--metrics-out`` (default ``evals/train_metrics.json``): training curve.

Numerical-stability notes
-------------------------
Mamba's selective scan is famously sensitive to precision (see the README's
"Troubleshooting" section). To avoid loss-NaN cascades during fine-tuning we:

* Always store master weights in **fp32**, regardless of ``--dtype`` flags.
* Use **AMP autocast** (fp16 on CUDA by default) so the forward is still fast.
* Wrap the optimizer in ``torch.amp.GradScaler`` when using fp16.
* **Skip** any optimizer step whose loss or grad-norm is non-finite, and abort
  if too many skips happen in a row.
* Compute the cross-entropy in fp32 even inside the autocast region.

Usage
-----
::

    # Tiny CPU smoke run (~5 min on a modern Mac CPU):
    python evals/train_esi.py

    # Recommended Colab T4 run:
    python evals/train_esi.py --device cuda --n-examples 4000 --epochs 2 \\
        --batch-size 2 --max-len 512 --lr 1e-4 --amp fp16

    # Side-by-side eval after training:
    python evals/evaluate_esi.py --esi-extras evals/esi_extras.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: E402


# ---------------------------------------------------------------------------


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def build_example(
    tokenizer,
    context: str,
    question: str,
    answer: str,
    max_len: int,
) -> Dict[str, torch.Tensor] | None:
    """Format one SQuAD example as ``(input_ids, question_ids, labels)``.

    ``labels`` are ``input_ids`` shifted-by-one with everything except the
    answer tokens masked to -100. Returns ``None`` if the example doesn't fit
    in ``max_len`` after truncating only the context.
    """
    prefix = "Context: "
    qmark = "\nQuestion: "
    amark = "\nAnswer:"
    ans = " " + answer.strip()

    # Token budget = max_len - room for everything except the context, leaving
    # the rest for the context. We never truncate the answer or question.
    pre_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    qm_ids = tokenizer(qmark, add_special_tokens=False).input_ids
    am_ids = tokenizer(amark, add_special_tokens=False).input_ids
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    a_ids = tokenizer(ans, add_special_tokens=False).input_ids
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids

    fixed = len(pre_ids) + len(qm_ids) + len(q_ids) + len(am_ids) + len(a_ids)
    budget = max_len - fixed
    if budget <= 16 or len(a_ids) == 0:
        return None
    ctx_ids = ctx_ids[:budget]

    input_ids = pre_ids + ctx_ids + qm_ids + q_ids + am_ids + a_ids
    # Loss labels: -100 everywhere except where we want to predict an answer
    # token. Standard "next-token" framing: a token at position ``t`` is the
    # target of the prediction made at position ``t-1``. We mark all answer
    # positions as targets.
    labels = [-100] * len(input_ids)
    ans_start = len(pre_ids) + len(ctx_ids) + len(qm_ids) + len(q_ids) + len(am_ids)
    for i in range(ans_start, len(input_ids)):
        labels[i] = input_ids[i]

    # Question fed to the ESI search is just the raw question text.
    question_ids = tokenizer(question, add_special_tokens=False).input_ids

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "question_ids": torch.tensor(question_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "answer": answer,
    }


def load_squad(tokenizer, n_examples: int, max_len: int, seed: int) -> List[dict]:
    """Load and pre-tokenize a SQuAD subset."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="train").shuffle(seed=seed)
    out: List[dict] = []
    for ex in ds:
        if not ex["answers"]["text"]:
            continue
        pkt = build_example(
            tokenizer,
            context=ex["context"],
            question=ex["question"],
            answer=ex["answers"]["text"][0],
            max_len=max_len,
        )
        if pkt is None:
            continue
        out.append(pkt)
        if len(out) >= n_examples:
            break
    return out


def pad_batch(batch: List[dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` / ``labels`` and ``question_ids`` separately."""
    max_in = max(b["input_ids"].size(0) for b in batch)
    max_q = max(b["question_ids"].size(0) for b in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_in), pad_id, dtype=torch.long)
    labels = torch.full((bs, max_in), -100, dtype=torch.long)
    question_ids = torch.full((bs, max_q), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        Lq = b["question_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        question_ids[i, :Lq] = b["question_ids"]
    return {"input_ids": input_ids, "labels": labels, "question_ids": question_ids}


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="state-spaces/mamba-130m")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    # NOTE: the master weights are always stored in fp32. ``--amp`` controls
    # whether the forward pass autocasts to a lower-precision dtype. Training
    # Mamba directly in fp16 is known to be unstable (see README precision
    # notes); fp32 master + fp16/bf16 autocast is the safe combination.
    p.add_argument(
        "--amp",
        default="auto",
        choices=["auto", "off", "fp16", "bf16"],
        help="Autocast mode for the forward pass. 'auto' = fp16 on CUDA, off elsewhere.",
    )
    p.add_argument("--n-examples", type=int, default=200, help="SQuAD train examples")
    p.add_argument("--max-len", type=int, default=512, help="Max combined seq length")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--max-nan-streak",
        type=int,
        default=10,
        help="Abort training if this many optimizer steps in a row are skipped due to NaN/Inf.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--esi-out", default=str(ROOT / "evals" / "esi_extras.pt"))
    p.add_argument("--metrics-out", default=str(ROOT / "evals" / "train_metrics.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = _device(args.device)

    # Decide the AMP autocast dtype. T4 (Turing) doesn't support bf16 — fall
    # back to fp16 there. CPU autocast is supported in recent torch but is
    # rarely worth it for this model, so default it to off.
    if args.amp == "auto":
        amp_mode = "fp16" if device.type == "cuda" else "off"
    else:
        amp_mode = args.amp
    if amp_mode == "bf16" and device.type == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        if major < 8:  # Ampere+ required for native bf16
            print("[train_esi] WARNING: bf16 requested but GPU is pre-Ampere; falling back to fp16.")
            amp_mode = "fp16"
    autocast_enabled = amp_mode in ("fp16", "bf16") and device.type == "cuda"
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode)
    use_grad_scaler = amp_mode == "fp16" and device.type == "cuda"
    print(f"[train_esi] device={device} master_dtype=float32 amp={amp_mode} grad_scaler={use_grad_scaler}")

    # --- model (always fp32 master weights) -----------------------------------
    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(
        args.model,
        device=str(device) if device.type != "mps" else "cpu",
        dtype=torch.float32,
    )
    if device.type == "mps":
        model = model.to(device)
    model.backbone.esi_enabled = True
    model.backbone.esi_top_k = args.top_k
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train_esi] loaded {n_total/1e6:.1f}M params in {time.time()-t0:.1f}s")

    # --- freeze backbone, unfreeze ESI extras ---------------------------------
    for p_ in model.parameters():
        p_.requires_grad_(False)
    esi_modules = {
        "question_embedding": model.question_embedding,
        "embedding_proj": model.backbone.embedding_proj,
        "injection_proj": model.backbone.injection_proj,
    }
    for mod in esi_modules.values():
        for p_ in mod.parameters():
            p_.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_esi] trainable params: {n_train:,} ({100*n_train/n_total:.2f}% of total)")

    # --- data -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"[train_esi] loading {args.n_examples} SQuAD examples (max_len={args.max_len})...")
    examples = load_squad(tokenizer, args.n_examples, args.max_len, args.seed)
    print(f"[train_esi] usable examples after filtering: {len(examples)}")

    # --- optimizer ------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    n_steps_per_epoch = max(1, math.ceil(len(examples) / args.batch_size))
    total_optim_steps = max(1, (n_steps_per_epoch * args.epochs) // args.grad_accum)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_optim_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # --- training loop --------------------------------------------------------
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    except TypeError:  # very old torch where GradScaler doesn't take device positional
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    def _should_log(step: int) -> bool:
        # Log every step for the first 10 optimizer steps, then every
        # ``args.log_every`` after that. Early logging is critical for
        # diagnosing instabilities.
        return step <= 10 or step % args.log_every == 0

    metrics: Dict[str, list] = {
        "step": [], "loss": [], "lr": [], "wallclock_s": [],
        "skipped": [], "grad_norm": [],
    }
    model.train()
    opt_step = 0
    micro_step = 0
    running_loss = 0.0
    running_tokens = 0
    nan_streak = 0
    n_skipped_total = 0
    abort_reason = None
    t_start = time.time()

    for epoch in range(args.epochs):
        if abort_reason:
            break
        rng = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(examples), generator=rng).tolist()
        for batch_start in range(0, len(order), args.batch_size):
            batch_idx = order[batch_start:batch_start + args.batch_size]
            batch = pad_batch([examples[i] for i in batch_idx], pad_id=pad_id)
            input_ids = batch["input_ids"].to(device)
            question_ids = batch["question_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(
                device_type=device.type if device.type in ("cuda", "cpu") else "cpu",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                out = model(input_ids, question_ids)
                logits = out.logits  # (B, L, V)
                # Cross-entropy is always done in fp32 for numerical stability
                # (autocast handles this automatically when targeting fp16/bf16,
                # but we make it explicit by upcasting).
                shift_logits = logits[:, :-1, :].float().contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                # Skip this micro-step entirely: don't accumulate gradients
                # from a NaN/Inf loss into a healthy partial sum.
                optim.zero_grad(set_to_none=True)
                nan_streak += 1
                n_skipped_total += 1
                print(f"[train_esi] WARNING: non-finite loss at micro_step={micro_step} (skipped)")
                if nan_streak >= args.max_nan_streak:
                    abort_reason = f"non-finite loss for {nan_streak} micro-steps in a row"
                    break
                micro_step += 1
                continue

            scaler.scale(loss / args.grad_accum).backward()
            n_tok = int((shift_labels != -100).sum().item())
            running_loss += loss_val * max(1, n_tok)
            running_tokens += max(1, n_tok)
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                cur_lr = lr_at(opt_step)
                for pg in optim.param_groups:
                    pg["lr"] = cur_lr

                # Unscale so we can clip in fp32 magnitudes regardless of AMP.
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
                grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                # ``scaler.step()`` internally skips when gradients are
                # non-finite (this is how AMP avoids corrupting fp32 master
                # weights). ``scaler.update()`` must follow to adjust the scale.
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                step_skipped = not math.isfinite(grad_norm_val)
                if step_skipped:
                    print(f"[train_esi] WARNING: non-finite grad norm at opt_step={opt_step+1} (update skipped)")
                    nan_streak += 1
                    n_skipped_total += 1
                else:
                    nan_streak = 0

                opt_step += 1

                if _should_log(opt_step):
                    avg_loss = running_loss / max(1, running_tokens)
                    wall = time.time() - t_start
                    flag = " [SKIPPED]" if step_skipped else ""
                    print(f"[train_esi] epoch={epoch} step={opt_step:>4} "
                          f"loss={avg_loss:.4f} grad_norm={grad_norm_val:.3f} "
                          f"lr={cur_lr:.2e} t={wall:.1f}s{flag}")
                    metrics["step"].append(opt_step)
                    metrics["loss"].append(avg_loss)
                    metrics["lr"].append(cur_lr)
                    metrics["wallclock_s"].append(wall)
                    metrics["skipped"].append(step_skipped)
                    metrics["grad_norm"].append(grad_norm_val)
                    running_loss = 0.0
                    running_tokens = 0

                if nan_streak >= args.max_nan_streak:
                    abort_reason = f"non-finite gradients for {nan_streak} optimizer steps in a row"
                    break

    if abort_reason is not None:
        print(f"[train_esi] ABORTED: {abort_reason}")
    optim.zero_grad(set_to_none=True)

    # --- save -----------------------------------------------------------------
    # Sanity check the trained weights before saving: a model full of NaN /
    # Inf is worse than no model at all, because the eval will silently load it
    # and produce garbage.
    bad_modules: List[str] = []
    for name, mod in {
        "question_embedding": model.question_embedding,
        "embedding_proj": model.backbone.embedding_proj,
        "injection_proj": model.backbone.injection_proj,
    }.items():
        for pname, ptens in mod.state_dict().items():
            if not torch.isfinite(ptens).all():
                bad_modules.append(f"{name}.{pname}")
                break
    if bad_modules:
        print(f"[train_esi] WARNING: non-finite weights detected in: {bad_modules}. "
              "Saving anyway, but the checkpoint will produce garbage at eval time. "
              "Lower the learning rate, raise --grad-clip, or use --amp off and rerun.")

    os.makedirs(os.path.dirname(args.esi_out), exist_ok=True)
    state = {
        "question_embedding": {k: v.detach().to(torch.float32).cpu()
                              for k, v in model.question_embedding.state_dict().items()},
        "embedding_proj":    {k: v.detach().to(torch.float32).cpu()
                              for k, v in model.backbone.embedding_proj.state_dict().items()},
        "injection_proj":    {k: v.detach().to(torch.float32).cpu()
                              for k, v in model.backbone.injection_proj.state_dict().items()},
        "config": vars(args),
        "non_finite_modules": bad_modules,
    }
    torch.save(state, args.esi_out)
    print(f"[train_esi] saved ESI extras -> {args.esi_out}")

    metrics["config"] = vars(args)
    metrics["n_total_params"] = n_total
    metrics["n_trainable_params"] = n_train
    metrics["n_examples"] = len(examples)
    metrics["total_seconds"] = time.time() - t_start
    metrics["n_skipped_steps"] = n_skipped_total
    metrics["abort_reason"] = abort_reason
    metrics["non_finite_modules"] = bad_modules
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[train_esi] wrote metrics -> {args.metrics_out}")
    print(f"[train_esi] DONE in {time.time()-t_start:.1f}s "
          f"(opt_steps={opt_step}, skipped={n_skipped_total})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
