"""End-to-end evaluation for MambaESI.

This script exercises the full MambaESI pipeline using the smallest publicly
available pretrained Mamba checkpoint (``state-spaces/mamba-130m``):

1. Loads the upstream Mamba weights into the ESI architecture (the ESI-only
   projection layers are randomly initialised — see ``[MambaESI]`` notice at
   load time).
2. Runs a smoke test to confirm the model produces plausible next-token
   predictions in vanilla (ESI-off) mode.
3. Computes token-level perplexity on a fixed-text held-out corpus, comparing
   ESI-off (= vanilla Mamba behaviour) against ESI-on (with the untrained
   projection layers in place).
4. Evaluates LAMBADA-style last-word accuracy on a small set of hand-curated
   long-distance dependency probes.
5. Inspects the embedding-search top-k retrieval: for several different
   questions against the same long context, prints out which context tokens
   the model selected.

Designed to run on CPU / MPS — no CUDA required. Patched modules in this fork
provide pure-PyTorch fallbacks for ``selective_scan`` and ``rms_norm``.

Usage:
    python evals/evaluate_esi.py
    python evals/evaluate_esi.py --model state-spaces/mamba-130m --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Make ``import mamba_ssm`` resolve when running from a fresh checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: E402


# A small but topically diverse held-out corpus used for perplexity scoring.
# Drawn from public-domain encyclopaedic prose so it stays under permissive
# distribution and exercises factual recall.
PPL_PASSAGES: List[str] = [
    "Mamba is a recently introduced state space model architecture that scales linearly "
    "with sequence length, making it attractive for long-context language modelling. "
    "Unlike Transformers, Mamba avoids the quadratic attention bottleneck by using a "
    "selective scan that propagates a small recurrent state through the sequence.",
    "The Great Wall of China was built over several centuries by successive Chinese "
    "dynasties to protect the northern borders. Construction began as early as the 7th "
    "century BC and continued through the Ming dynasty, resulting in a network of walls "
    "spanning thousands of kilometres across the country.",
    "Photosynthesis is the biological process by which green plants, algae, and certain "
    "bacteria convert light energy, usually from the sun, into chemical energy stored in "
    "glucose. Carbon dioxide and water are consumed in the process and oxygen is released "
    "as a by-product, sustaining most aerobic life on Earth.",
    "In computer science, a hash table is a data structure that maps keys to values "
    "using a hash function. Average-case operations run in constant time, but the worst "
    "case can degrade to linear time if many keys collide. Good hash functions and "
    "appropriate resizing strategies keep the load factor low and lookups fast.",
]


# Hand-built LAMBADA-style examples. The model is given the context and asked to
# predict the final word. We score the first sub-token of the gold word
# (consistent with the standard LAMBADA convention for sub-word tokenisers).
@dataclass
class LambadaExample:
    context: str
    target: str   # The full final word (used for display).
    question: str  # The "question" presented to the ESI mechanism.


LAMBADA_EXAMPLES: List[LambadaExample] = [
    LambadaExample(
        context=(
            "Sarah had been planning her trip to Paris for months. She finally booked the "
            "flights, packed her bags, and waved goodbye to her family. When her plane "
            "landed in"
        ),
        target=" Paris",
        question="Where did Sarah's plane land?",
    ),
    LambadaExample(
        context=(
            "The detective looked carefully at the muddy footprints near the back door. "
            "They were clearly made by heavy boots, far too large for the victim. The "
            "detective concluded that the crime had been committed by a"
        ),
        target=" man",
        question="What did the detective conclude about the criminal?",
    ),
    LambadaExample(
        context=(
            "The chef sprinkled the dish with fresh basil, drizzled it with olive oil, "
            "and topped it with shaved parmesan. To finish the classic Italian dish, all "
            "that remained was a sprinkle of"
        ),
        target=" salt",
        question="What was the final ingredient added to the dish?",
    ),
    LambadaExample(
        context=(
            "Mary loved sailing more than anything else in the world. Every weekend she "
            "took her small wooden boat onto the lake and watched the sun set across the "
            "water. Today she could not wait to get back on her"
        ),
        target=" boat",
        question="What did Mary look forward to?",
    ),
    LambadaExample(
        context=(
            "The young pianist had practised the concerto for nearly a year. As the "
            "orchestra began the opening bars, she took a deep breath, smiled at the "
            "conductor, and placed her hands on the"
        ),
        target=" piano",
        question="What instrument was the young musician about to play?",
    ),
    LambadaExample(
        context=(
            "After hiking for hours up the steep trail, Tom finally reached the summit. "
            "He sat down on a smooth rock, took out his water bottle, and admired the "
            "view from the top of the"
        ),
        target=" mountain",
        question="Where did Tom rest after his hike?",
    ),
    LambadaExample(
        context=(
            "The astronaut floated past the small porthole and gazed out at the planet "
            "below. Continents and oceans drifted slowly beneath her, and she could even "
            "make out the green band of the Amazon. She was orbiting the planet"
        ),
        target=" Earth",
        question="Which planet was the astronaut orbiting?",
    ),
    LambadaExample(
        context=(
            "The librarian carefully placed the ancient manuscript back onto the shelf. "
            "It was the oldest item in the collection and dated from the fifteenth "
            "century. Visitors were never permitted to handle the rare"
        ),
        target=" book",
        question="What were visitors not allowed to handle?",
    ),
]


# Context used to demonstrate the embedding-search retrieval. Each question
# focuses on a different fact embedded in the same passage.
RETRIEVAL_CONTEXT = (
    "Jane is a 32-year-old marine biologist working in Australia. She specialises in "
    "studying coral reefs and lives in a small wooden cottage near Cairns. Her favourite "
    "research subject is the clownfish, which she has been observing for five years. "
    "On weekends she paints watercolours and bakes sourdough bread."
)

RETRIEVAL_QUESTIONS = [
    "What is Jane's profession?",
    "Where does Jane live?",
    "Which animal does Jane study?",
    "What does Jane do on weekends?",
]


# ---------- helpers ----------------------------------------------------------


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, question: str | None, device) -> float:
    """Token-level perplexity of ``text`` under the model.

    Computed as ``exp(mean(NLL))`` over the next-token loss for every non-first
    token in the passage. When ``question`` is None, the ESI mechanism is
    bypassed.
    """
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if enc.shape[1] < 2:
        return float("nan")

    if question is None:
        model.backbone.esi_enabled = False
        qids = None
    else:
        model.backbone.esi_enabled = True
        qids = tokenizer(question, return_tensors="pt").input_ids.to(device)

    logits = model(enc, qids).logits  # (1, L, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = enc[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())


@torch.no_grad()
def lambada_accuracy(model, tokenizer, examples: List[LambadaExample], use_esi: bool, device) -> Tuple[float, List[dict]]:
    """LAMBADA-style accuracy on the first sub-token of the gold final word."""
    model.backbone.esi_enabled = use_esi
    correct = 0
    rows = []
    for ex in examples:
        ctx_ids = tokenizer(ex.context, return_tensors="pt").input_ids.to(device)
        tgt_ids = tokenizer(ex.target, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        gold = tgt_ids[0]
        qids = tokenizer(ex.question, return_tensors="pt").input_ids.to(device) if use_esi else None
        logits = model(ctx_ids, qids).logits  # (1, L, V)
        pred = int(logits[0, -1].argmax().item())
        is_correct = pred == gold
        correct += int(is_correct)
        rows.append({
            "question": ex.question,
            "gold": tokenizer.decode([gold]),
            "pred": tokenizer.decode([pred]),
            "correct": is_correct,
        })
    return correct / len(examples), rows


@torch.no_grad()
def demo_retrieval(model, tokenizer, context: str, questions: List[str], device, top_k: int = 5):
    """Run ESI on the same context with several questions and report top-k tokens."""
    model.backbone.esi_enabled = True
    model.backbone.esi_top_k = top_k

    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    ctx_token_strs = [tokenizer.decode([t]) for t in ctx_ids[0].tolist()]

    results = []
    for q in questions:
        qids = tokenizer(q, return_tensors="pt").input_ids.to(device)
        out = model(ctx_ids, qids, return_top_indices=True)
        top = out.top_indices[0].tolist()
        top_tokens = [ctx_token_strs[i] for i in top]
        results.append({"question": q, "top_indices": top, "top_tokens": top_tokens})
    return results, ctx_token_strs


# ---------- main -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="state-spaces/mamba-130m", help="HF model id or local path")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b", help="Tokenizer compatible with Mamba (defaults to GPT-NeoX-20B BPE).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", default=str(ROOT / "evals" / "results.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--esi-extras",
        default=None,
        help=(
            "Optional path to an ESI-extras state dict produced by "
            "evals/train_esi.py. When supplied, the question_embedding, "
            "embedding_proj, and injection_proj layers are overwritten with "
            "the trained weights before running the eval."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[MambaESI eval] device={device}, dtype={dtype}, model={args.model}")
    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(args.model, device=str(device) if device.type != "mps" else "cpu", dtype=dtype)
    if device.type == "mps":
        model = model.to(device)
    model.eval()
    model.backbone.esi_top_k = args.top_k
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MambaESI eval] loaded {n_params/1e6:.1f}M params in {time.time()-t0:.1f}s")

    esi_extras_loaded = False
    if args.esi_extras:
        extras_path = Path(args.esi_extras)
        if not extras_path.exists():
            raise FileNotFoundError(f"--esi-extras file not found: {extras_path}")
        try:
            state = torch.load(extras_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(extras_path, map_location=device)
        model.question_embedding.load_state_dict(state["question_embedding"])
        model.backbone.embedding_proj.load_state_dict(state["embedding_proj"])
        model.backbone.injection_proj.load_state_dict(state["injection_proj"])
        esi_extras_loaded = True
        print(f"[MambaESI eval] loaded trained ESI extras from {extras_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- 1. quick smoke generation ------------------------------------------
    prompt = "The Eiffel Tower is located in"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    model.backbone.esi_enabled = False
    with torch.no_grad():
        nxt = int(model(ids).logits[0, -1].argmax().item())
    smoke_pred = tokenizer.decode([nxt])
    print(f"[smoke] vanilla Mamba next token after {prompt!r}: {smoke_pred!r}")

    # --- 2. perplexity --------------------------------------------------------
    print("\n[Perplexity, lower is better]")
    ppl_off, ppl_on = [], []
    ppl_rows = []
    for i, passage in enumerate(PPL_PASSAGES):
        # Use the first sentence as the implicit "question" so the ESI mechanism
        # has something semantically related to the passage.
        question = passage.split(".")[0] + "."
        pp_off = compute_perplexity(model, tokenizer, passage, None, device)
        pp_on = compute_perplexity(model, tokenizer, passage, question, device)
        ppl_off.append(pp_off)
        ppl_on.append(pp_on)
        ppl_rows.append({"idx": i, "esi_off": pp_off, "esi_on": pp_on, "n_chars": len(passage)})
        print(f"  passage {i}: esi_off={pp_off:7.2f}   esi_on={pp_on:7.2f}")
    mean_off = sum(ppl_off) / len(ppl_off)
    mean_on = sum(ppl_on) / len(ppl_on)
    print(f"  mean       : esi_off={mean_off:7.2f}   esi_on={mean_on:7.2f}")

    # --- 3. LAMBADA-style accuracy -------------------------------------------
    print("\n[LAMBADA-style last-word accuracy, higher is better]")
    acc_off, rows_off = lambada_accuracy(model, tokenizer, LAMBADA_EXAMPLES, use_esi=False, device=device)
    acc_on, rows_on = lambada_accuracy(model, tokenizer, LAMBADA_EXAMPLES, use_esi=True, device=device)
    print(f"  esi_off: {acc_off*100:5.1f}%   esi_on: {acc_on*100:5.1f}%   (n={len(LAMBADA_EXAMPLES)})")
    for i, (r_off, r_on) in enumerate(zip(rows_off, rows_on)):
        print(
            f"  [{i}] gold={r_off['gold']!r:>10}   "
            f"off={r_off['pred']!r:>10}({'Y' if r_off['correct'] else 'n'})   "
            f"on={r_on['pred']!r:>10}({'Y' if r_on['correct'] else 'n'})"
        )

    # --- 4. retrieval demo ----------------------------------------------------
    print("\n[ESI retrieval: top-5 tokens selected per question, against shared context]")
    retrieval_rows, _ = demo_retrieval(model, tokenizer, RETRIEVAL_CONTEXT, RETRIEVAL_QUESTIONS, device, top_k=args.top_k)
    for r in retrieval_rows:
        print(f"  Q: {r['question']}")
        print(f"     top tokens: {r['top_tokens']}")

    # --- 5. dump JSON ---------------------------------------------------------
    summary = {
        "config": {
            "model": args.model,
            "device": str(device),
            "dtype": str(dtype),
            "n_params": n_params,
            "esi_top_k": args.top_k,
            "esi_extras_loaded": esi_extras_loaded,
            "esi_extras_path": args.esi_extras,
        },
        "smoke": {"prompt": prompt, "next_token_vanilla": smoke_pred},
        "perplexity": {
            "per_passage": ppl_rows,
            "mean_esi_off": mean_off,
            "mean_esi_on": mean_on,
        },
        "lambada": {
            "accuracy_esi_off": acc_off,
            "accuracy_esi_on": acc_on,
            "rows_esi_off": rows_off,
            "rows_esi_on": rows_on,
        },
        "retrieval": retrieval_rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[MambaESI eval] wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
