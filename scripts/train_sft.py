"""Supervised fine-tuning (SFT) of Qwen2.5-0.5B on single-turn UltraChat examples.

    # 2-minute sanity check (no checkpoints saved)
    python -u scripts/train_sft.py --output_dir outputs/sft_sanity \
        --max_steps 5 --val_size 32 --eval_every 5 --no_save

    # full run (guide configuration)
    python -u scripts/train_sft.py --output_dir outputs/sft

What it does
  * Data: HuggingFaceH4/ultrachat_200k. Each conversation becomes ONE single-turn
    example: its first user message (instruction) and the first assistant reply
    (response). By default only examples whose formatted prompt + response + EOS
    fit in --max_length are kept, so every training response ends with EOS and the
    model learns when to stop. Selection uses a seeded shuffle and is recorded in
    <output_dir>/data_manifest.json (dataset indices + prompt_ids).
  * Loss: response-only token-mean NLL (alignment.losses.sft_loss), prompt and
    padding masked with -100.
  * Optimisation: AdamW (no weight decay on biases/norm weights), linear warmup then
    cosine decay to 0, gradient accumulation, gradient clipping, fp16 autocast with
    fp32 master weights and a GradScaler (T4 GPUs have no fast bf16).
  * Validation loss (token-weighted over the whole validation set) at step 0, every
    --eval_every optimizer steps and at the end. Saves `best` (lowest validation
    loss) and `final` checkpoints. Metrics go to <output_dir>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.batching import build_lm_batch  # noqa: E402
from alignment.logprobs import batch_logprobs  # noqa: E402
from alignment.losses import sft_loss  # noqa: E402
from alignment.prompting import SYSTEM_PROMPT, format_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Response-only SFT for a causal LM.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--train_split", default="train_sft")
    p.add_argument("--val_split", default="test_sft")
    p.add_argument("--train_size", type=int, default=30_000)
    p.add_argument("--val_size", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--allow_truncated", action="store_true",
                   help="also keep examples longer than --max_length (they lose their EOS)")
    p.add_argument("--batch_size", type=int, default=2, help="per-device micro-batch")
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None, help="cap optimizer steps (sanity runs)")
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gradient_checkpointing", action="store_true", help="trade speed for memory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--no_save", action="store_true", help="skip writing model checkpoints")
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def single_turn(messages) -> tuple[str, str] | None:
    """First user message and the assistant reply that follows it, or None."""
    for i, message in enumerate(messages[:-1]):
        if message.get("role") == "user":
            reply = messages[i + 1]
            if reply.get("role") == "assistant":
                instruction, response = message.get("content", ""), reply.get("content", "")
                if instruction.strip() and response.strip():
                    return instruction, response
            return None
    return None


def select_examples(dataset, n, tokenizer, max_length, seed, allow_truncated, chunk=2000):
    """Seeded shuffle of the split, keep the first ``n`` usable single-turn examples."""
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    eos_extra = 1 if tokenizer.eos_token_id is not None else 0
    stats = {"split_size": len(dataset), "rows_seen": 0, "no_single_turn": 0,
             "too_long_skipped": 0, "kept": 0}
    selected = []
    for start in range(0, len(order), chunk):
        indices = order[start : start + chunk]
        rows = dataset[indices]
        prompt_ids = rows.get("prompt_id", [None] * len(indices))
        candidates = []
        for index, messages, prompt_id in zip(indices, rows["messages"], prompt_ids):
            stats["rows_seen"] += 1
            pair = single_turn(messages)
            if pair is None:
                stats["no_single_turn"] += 1
                continue
            candidates.append((index, prompt_id, *pair))
        if not candidates:
            continue
        prompt_tokens = tokenizer([format_prompt(c[2]) for c in candidates], add_special_tokens=False)["input_ids"]
        response_tokens = tokenizer([c[3] for c in candidates], add_special_tokens=False)["input_ids"]
        for (index, prompt_id, instruction, response), p_ids, r_ids in zip(
            candidates, prompt_tokens, response_tokens
        ):
            if len(p_ids) + len(r_ids) + eos_extra > max_length and not allow_truncated:
                stats["too_long_skipped"] += 1
                continue
            selected.append({"index": index, "prompt_id": prompt_id,
                             "instruction": instruction, "response": response})
            if len(selected) >= n:
                stats["kept"] = len(selected)
                return selected, stats
    stats["kept"] = len(selected)
    print(f"WARNING: only {len(selected)} usable examples found (wanted {n})")
    return selected, stats


def make_batch(tokenizer, examples, max_length):
    return build_lm_batch(
        tokenizer,
        [format_prompt(e["instruction"]) for e in examples],
        [e["response"] for e in examples],
        max_length,
    )


# ------------------------------------------------------------------------- training


@torch.no_grad()
def evaluate(model, tokenizer, examples, args, use_amp) -> float:
    """Token-weighted mean response NLL over the whole validation set."""
    was_training = model.training
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for start in range(0, len(examples), args.eval_batch_size):
        batch = make_batch(tokenizer, examples[start : start + args.eval_batch_size], args.max_length)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logps, counts = batch_logprobs(model, batch)
        total_logp += logps.sum().item()
        total_tokens += int(counts.sum().item())
    model.train(was_training)
    return -total_logp / max(total_tokens, 1)


def save_checkpoint(model, tokenizer, directory: Path, info: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / "checkpoint_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading and selecting data ...", flush=True)
    train_split = load_dataset(args.dataset, split=args.train_split)
    val_split = load_dataset(args.dataset, split=args.val_split)
    train, train_stats = select_examples(train_split, args.train_size, tokenizer, args.max_length,
                                         args.seed, args.allow_truncated)
    val, val_stats = select_examples(val_split, args.val_size, tokenizer, args.max_length,
                                     args.seed, args.allow_truncated)
    manifest = {
        "dataset": args.dataset, "seed": args.seed, "max_length": args.max_length,
        "allow_truncated": args.allow_truncated,
        "train": {"split": args.train_split, **train_stats,
                  "indices": [e["index"] for e in train], "prompt_ids": [e["prompt_id"] for e in train]},
        "val": {"split": args.val_split, **val_stats,
                "indices": [e["index"] for e in val], "prompt_ids": [e["prompt_id"] for e in val]},
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest))
    print(f"train: {train_stats}\nval:   {val_stats}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() or name.endswith("bias") else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:  # step = number of optimizer updates already done
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**vars(args), "device": device, "precision": "fp16 autocast + fp32 weights" if use_amp else "fp32",
              "effective_batch_size": args.batch_size * args.grad_accum, "total_steps": total_steps,
              "warmup_steps": warmup_steps, "num_train": len(train), "num_val": len(val),
              "system_prompt": SYSTEM_PROMPT, "loss_reduction": "token-mean over response tokens",
              "optimizer": "AdamW(betas=(0.9,0.999), eps=1e-8)", "schedule": "linear warmup + cosine to 0",
              "torch": torch.__version__}
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    metrics_file = open(out / "metrics.jsonl", "w")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    start_time = time.time()
    val_loss = evaluate(model, tokenizer, val, args, use_amp)
    log({"type": "val", "step": 0, "val_loss": val_loss, "val_ppl": math.exp(val_loss)})
    print(f"step 0 | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}", flush=True)
    best_val, best_step = float("inf"), None

    global_step = 0
    done = False
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        micro_batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for group_start in range(0, len(micro_batches), args.grad_accum):
            group = micro_batches[group_start : group_start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in group:
                batch = make_batch(tokenizer, [train[i] for i in micro], args.max_length)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = sft_loss(model, batch)
                scaler.scale(loss / len(group)).backward()
                loss_sum += loss.item()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                record = {"type": "train", "step": global_step, "epoch": epoch,
                          "train_loss": loss_sum / len(group), "lr": scheduler.get_last_lr()[0],
                          "grad_norm": grad_norm, "loss_scale": scaler.get_scale(),
                          "elapsed_min": (time.time() - start_time) / 60}
                log(record)
                print(f"step {global_step}/{total_steps} | loss {record['train_loss']:.4f} | "
                      f"lr {record['lr']:.2e} | gnorm {grad_norm:.3f} | "
                      f"{record['elapsed_min']:.1f} min", flush=True)

            if global_step % args.eval_every == 0 or global_step == total_steps:
                val_loss = evaluate(model, tokenizer, val, args, use_amp)
                log({"type": "val", "step": global_step, "val_loss": val_loss, "val_ppl": math.exp(val_loss)})
                print(f"step {global_step} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}",
                      flush=True)
                if val_loss < best_val:
                    best_val, best_step = val_loss, global_step
                    if not args.no_save:
                        save_checkpoint(model, tokenizer, out / "best",
                                        {"step": global_step, "val_loss": val_loss})
                        print(f"  saved best checkpoint (step {global_step})", flush=True)

            if global_step >= total_steps:
                done = True
                break
        if done:
            break

    if not args.no_save:
        save_checkpoint(model, tokenizer, out / "final", {"step": global_step, "val_loss": val_loss})
    summary = {"total_steps": global_step, "best_step": best_step, "best_val_loss": best_val,
               "final_val_loss": val_loss, "minutes": (time.time() - start_time) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    metrics_file.close()
    print("done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
