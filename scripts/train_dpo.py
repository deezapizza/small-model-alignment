"""From-scratch DPO on top of the SFT checkpoint (no TRL, no HF Trainer).

    # 2-minute sanity check (no checkpoints saved)
    python -u scripts/train_dpo.py --dataset hh --sft_model outputs/sft/best \
        --output_dir outputs/dpo_hh_sanity --max_steps 3 --val_size 16 --eval_every 3 --no_save

    # full runs
    python -u scripts/train_dpo.py --dataset hh            --sft_model outputs/sft/best --output_dir outputs/dpo_hh
    python -u scripts/train_dpo.py --dataset ultrafeedback --sft_model outputs/sft/best --output_dir outputs/dpo_uf

What it does
  * Policy and frozen reference both start from the SFT checkpoint. The policy
    trains on cuda:0 (fp32 weights, fp16 autocast, GradScaler); the reference sits
    on cuda:1 when a second GPU exists (same dtype/autocast, so at step 0 the two
    give identical log-probs and reward accuracy is exactly 0), always in eval mode
    with requires_grad=False and scored under torch.no_grad (inside dpo_loss).
  * Data (alignment.dpo_data): single-turn HH-RLHF pairs or UltraFeedback-binarized
    pairs, seeded shuffle, first --train_size usable pairs from the train split and
    --val_size from the test split. Pairs whose formatted prompt is longer than
    --max_prompt_length are skipped; responses are right-truncated to --max_length.
  * Loss: alignment.losses.dpo_loss (beta * difference of policy/reference log-ratio
    gaps, -logsigmoid, batch mean). Metrics: reward accuracy (strict r_w > r_l),
    reward margin, chosen/rejected rewards, mean chosen/rejected log-prob per token.
  * Optimiser: RMSprop (as in the DPO paper) or AdamW, linear warmup then constant LR.
  * Validation on the whole held-out set at step 0, every --eval_every steps and at
    the end; reference val log-probs are computed once and cached. Saves `best`
    (highest val reward accuracy, ties -> lower val loss) and `final`.
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
from alignment.batching import build_dpo_batch  # noqa: E402
from alignment.dpo_data import DATASETS, parse_row  # noqa: E402
from alignment.logprobs import batch_logprobs  # noqa: E402
from alignment.losses import dpo_loss, dpo_loss_from_logps, dpo_reward_metrics  # noqa: E402
from alignment.prompting import SYSTEM_PROMPT, format_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="From-scratch DPO.")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--sft_model", required=True, help="SFT checkpoint dir (policy init AND reference)")
    p.add_argument("--train_size", type=int, default=15_000)
    p.add_argument("--val_size", type=int, default=1_000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="linear warmup, then constant LR")
    p.add_argument("--batch_size", type=int, default=1, help="pairs per micro-batch")
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_save", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def select_pairs(name, split, n, tokenizer, max_prompt_length, seed, chunk=2000):
    order = list(range(len(split)))
    random.Random(seed).shuffle(order)
    stats = {"split_size": len(split), "rows_seen": 0, "unusable": 0, "prompt_too_long": 0, "kept": 0}
    pairs = []
    for start in range(0, len(order), chunk):
        indices = order[start : start + chunk]
        rows = split[indices]
        keys = list(rows.keys())
        candidates = []
        for j, index in enumerate(indices):
            stats["rows_seen"] += 1
            parsed = parse_row(name, {k: rows[k][j] for k in keys})
            if parsed is None:
                stats["unusable"] += 1
                continue
            candidates.append((index, *parsed))
        if not candidates:
            continue
        lengths = [len(ids) for ids in tokenizer([format_prompt(c[1]) for c in candidates],
                                                 add_special_tokens=False)["input_ids"]]
        for (index, instruction, chosen, rejected), length in zip(candidates, lengths):
            if length > max_prompt_length:
                stats["prompt_too_long"] += 1
                continue
            pairs.append({"index": index, "instruction": instruction, "chosen": chosen, "rejected": rejected})
            if len(pairs) >= n:
                stats["kept"] = len(pairs)
                return pairs, stats
    stats["kept"] = len(pairs)
    print(f"WARNING: only {len(pairs)} usable pairs (wanted {n})")
    return pairs, stats


def columns(pairs):
    return ([format_prompt(p["instruction"]) for p in pairs],
            [p["chosen"] for p in pairs], [p["rejected"] for p in pairs])


# ------------------------------------------------------------------------ scoring


@torch.no_grad()
def score_pairs(model, tokenizer, pairs, args, use_amp):
    """Summed response log-probs (and token counts) for chosen and rejected, over all pairs."""
    was_training = model.training
    model.eval()
    out = {"chosen": [], "rejected": [], "chosen_tokens": [], "rejected_tokens": []}
    for start in range(0, len(pairs), args.eval_batch_size):
        prompts, chosen, rejected = columns(pairs[start : start + args.eval_batch_size])
        batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length, args.max_prompt_length)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for side in ("chosen", "rejected"):
                logps, counts = batch_logprobs(model, batch[side])
                out[side].append(logps.float().cpu())
                out[side + "_tokens"].append(counts.cpu())
    model.train(was_training)
    return {k: torch.cat(v) for k, v in out.items()}


def validation_metrics(policy_scores, reference_scores, beta) -> dict:
    args = (policy_scores["chosen"], policy_scores["rejected"],
            reference_scores["chosen"], reference_scores["rejected"])
    metrics = {"val_loss": dpo_loss_from_logps(*args, beta).item()}
    metrics.update({f"val_{k}": v for k, v in dpo_reward_metrics(*args, beta).items()})
    for side in ("chosen", "rejected"):
        per_token = policy_scores[side] / policy_scores[side + "_tokens"].clamp_min(1)
        metrics[f"val_{side}_logp_per_token"] = per_token.mean().item()
    return metrics


def save_checkpoint(model, tokenizer, directory: Path, info: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / "checkpoint_info.json").write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------------- training


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    policy_device = "cuda:0" if n_gpu else "cpu"
    reference_device = "cuda:1" if n_gpu >= 2 else policy_device
    use_amp = n_gpu > 0

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = DATASETS[args.dataset]
    print(f"loading {spec['hf_id']} ...", flush=True)
    train_split = load_dataset(spec["hf_id"], split=spec["train_split"])
    val_split = load_dataset(spec["hf_id"], split=spec["val_split"])
    train, train_stats = select_pairs(args.dataset, train_split, args.train_size, tokenizer,
                                      args.max_prompt_length, args.seed)
    val, val_stats = select_pairs(args.dataset, val_split, args.val_size, tokenizer,
                                  args.max_prompt_length, args.seed)
    manifest = {"dataset": spec["hf_id"], "seed": args.seed, "max_prompt_length": args.max_prompt_length,
                "train": {"split": spec["train_split"], **train_stats, "indices": [p["index"] for p in train]},
                "val": {"split": spec["val_split"], **val_stats, "indices": [p["index"] for p in val]}}
    (out / "data_manifest.json").write_text(json.dumps(manifest))
    print(f"train: {train_stats}\nval:   {val_stats}", flush=True)
    print("example pair:", json.dumps({k: v[:200] for k, v in train[0].items() if k != "index"}, indent=1))

    policy = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(args.sft_model, torch_dtype=torch.float32).to(reference_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    pairs_per_step = args.batch_size * args.grad_accum
    total_steps = math.ceil(len(train) / pairs_per_step)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**vars(args), "hf_dataset": spec["hf_id"], "policy_device": policy_device,
              "reference_device": reference_device, "precision": "fp16 autocast + fp32 weights (policy and reference)",
              "effective_batch_size": pairs_per_step, "total_steps": total_steps, "warmup_steps": warmup_steps,
              "schedule": "linear warmup then constant", "num_train": len(train), "num_val": len(val),
              "system_prompt": SYSTEM_PROMPT, "torch": torch.__version__}
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    metrics_file = open(out / "metrics.jsonl", "w")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    start_time = time.time()
    print("scoring validation set with the reference (cached) ...", flush=True)
    reference_val = score_pairs(reference, tokenizer, val, args, use_amp)
    best = {"acc": -1.0, "loss": float("inf"), "step": None}

    def run_validation(step: int) -> dict:
        metrics = validation_metrics(score_pairs(policy, tokenizer, val, args, use_amp), reference_val, args.beta)
        log({"type": "val", "step": step, **metrics, "elapsed_min": (time.time() - start_time) / 60})
        print(f"step {step} | val_loss {metrics['val_loss']:.4f} | val_acc {metrics['val_reward_accuracy']:.3f} | "
              f"val_margin {metrics['val_reward_margin']:.4f} | chosen_logp/tok "
              f"{metrics['val_chosen_logp_per_token']:.3f}", flush=True)
        better = (metrics["val_reward_accuracy"] > best["acc"]
                  or (metrics["val_reward_accuracy"] == best["acc"] and metrics["val_loss"] < best["loss"]))
        if step > 0 and better:
            best.update(acc=metrics["val_reward_accuracy"], loss=metrics["val_loss"], step=step)
            if not args.no_save:
                save_checkpoint(policy, tokenizer, out / "best", {"step": step, **metrics})
                print(f"  saved best checkpoint (step {step})", flush=True)
        return metrics

    val_metrics = run_validation(0)

    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    keys = ("reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward")
    for step in range(1, total_steps + 1):
        group = order[(step - 1) * pairs_per_step : step * pairs_per_step]
        if not group:
            break
        optimizer.zero_grad(set_to_none=True)
        loss_sum, sums = 0.0, dict.fromkeys(keys, 0.0)
        for start in range(0, len(group), args.batch_size):
            micro = [train[i] for i in group[start : start + args.batch_size]]
            prompts, chosen, rejected = columns(micro)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, metrics = dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected,
                                         beta=args.beta, max_length=args.max_length,
                                         max_prompt_length=args.max_prompt_length, return_metrics=True)
            weight = len(micro) / len(group)  # exact mean over the pairs of this optimizer step
            scaler.scale(loss * weight).backward()
            loss_sum += loss.item() * weight
            for k in keys:
                sums[k] += metrics[k] * weight
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            record = {"type": "train", "step": step, "train_loss": loss_sum, **{f"train_{k}": v for k, v in sums.items()},
                      "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm, "loss_scale": scaler.get_scale(),
                      "elapsed_min": (time.time() - start_time) / 60}
            log(record)
            print(f"step {step}/{total_steps} | loss {loss_sum:.4f} | acc {sums['reward_accuracy']:.3f} | "
                  f"margin {sums['reward_margin']:.4f} | lr {record['lr']:.2e} | gnorm {grad_norm:.2f} | "
                  f"{record['elapsed_min']:.1f} min", flush=True)

        if step % args.eval_every == 0 or step == total_steps:
            val_metrics = run_validation(step)

    if not args.no_save:
        save_checkpoint(policy, tokenizer, out / "final", {"step": total_steps, **val_metrics})
    summary = {"total_steps": total_steps, "best_step": best["step"], "best_val_reward_accuracy": best["acc"],
               "best_val_loss": best["loss"], "final": val_metrics, "minutes": (time.time() - start_time) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    metrics_file.close()
    print("done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
