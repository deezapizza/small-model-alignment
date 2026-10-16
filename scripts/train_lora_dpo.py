"""Task 5: persona DPO with TRL's DPOTrainer + PEFT/LoRA on SmolLM2-360M-Instruct.

    pip install -q trl peft          # on Kaggle, after the repo install
    python -u scripts/train_lora_dpo.py --output_dir outputs/lora_dpo_persona

What it does
  * Data: data/persona_lora/ramsay_style_preferences.json (prompt / chosen / rejected),
    converted to TRL's *conversational* preference format
        prompt   = [{"role": "user", "content": ...}]
        chosen   = [{"role": "assistant", "content": ...}]
        rejected = [{"role": "assistant", "content": ...}]
    so DPOTrainer applies SmolLM2's own chat template. A seeded split holds out
    --val_size pairs for validation.
  * Model: LoRA adapters (PEFT) on all attention and MLP projections; the base
    weights stay frozen. No separate reference model is loaded: with a PEFT model
    DPOTrainer uses the same network with the adapter disabled as the reference.
  * Logs DPO loss / reward accuracy / reward margin (train + validation) to
    <output_dir>/metrics.jsonl in the same format as scripts/train_dpo.py, so
    scripts/plot_dpo.py can plot it. Saves the adapter (<output_dir>/adapter) and a
    merged full model (<output_dir>/merged) that scripts/generate.py can load.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 is plenty; avoids DataParallel
os.environ.setdefault("WANDB_DISABLED", "true")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--val_size", type=int, default=20)
    p.add_argument("--output_dir", default="outputs/lora_dpo_persona")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--eval_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1, help="sanity runs")
    return p.parse_args()


def conversational(row: dict) -> dict:
    return {"prompt": [{"role": "user", "content": row["prompt"].strip()}],
            "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
            "rejected": [{"role": "assistant", "content": row["rejected"].strip()}]}


def only_supported(cls, kwargs: dict) -> dict:
    """Keep the kwargs this TRL version's DPOConfig accepts (field names change across releases)."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in kwargs if k not in names)
    if dropped:
        print(f"note: this TRL version has no DPOConfig fields {dropped}; ignoring them")
    return {k: v for k, v in kwargs.items() if k in names}


def to_metrics(log_history: list[dict]) -> list[dict]:
    """Map Trainer log records to the metrics.jsonl schema used by plot_dpo.py."""
    out = []
    for rec in log_history:
        step = rec.get("step")
        if "eval_loss" in rec:
            out.append({"type": "val", "step": step, "val_loss": rec["eval_loss"],
                        "val_reward_accuracy": rec.get("eval_rewards/accuracies"),
                        "val_reward_margin": rec.get("eval_rewards/margins"),
                        "val_chosen_reward": rec.get("eval_rewards/chosen"),
                        "val_rejected_reward": rec.get("eval_rewards/rejected"),
                        "epoch": rec.get("epoch")})
        elif "loss" in rec and "rewards/accuracies" in rec:
            out.append({"type": "train", "step": step, "train_loss": rec["loss"],
                        "train_reward_accuracy": rec["rewards/accuracies"],
                        "train_reward_margin": rec.get("rewards/margins"),
                        "lr": rec.get("learning_rate"), "grad_norm": rec.get("grad_norm"),
                        "epoch": rec.get("epoch")})
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import peft
    import transformers
    import trl
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "trl": trl.__version__, "peft": peft.__version__}
    print(versions, flush=True)

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    val_idx, train_idx = sorted(order[: args.val_size]), sorted(order[args.val_size :])
    train_ds = Dataset.from_list([conversational(rows[i]) for i in train_idx])
    val_ds = Dataset.from_list([conversational(rows[i]) for i in val_idx])
    (out / "data_split.json").write_text(json.dumps({"train_indices": train_idx, "val_indices": val_idx}))
    print(f"train pairs {len(train_ds)} | val pairs {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    example = tokenizer.apply_chat_template(train_ds[0]["prompt"], tokenize=False, add_generation_prompt=True)
    print("formatted prompt example:\n" + example)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.target_modules.split(","), bias="none", task_type="CAUSAL_LM")
    use_fp16 = torch.cuda.is_available()
    # warmup as an explicit step count: newer TRL/transformers releases dropped `warmup_ratio`
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, round(args.warmup_ratio * planned_steps))
    print(f"planned optimizer steps {planned_steps} | warmup steps {warmup_steps}")
    config_kwargs = dict(
        output_dir=str(out / "trainer"), beta=args.beta, loss_type="sigmoid",
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine",
        warmup_steps=warmup_steps, weight_decay=0.0, max_grad_norm=1.0,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        logging_steps=1, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="no",
        fp16=use_fp16, bf16=False, report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    dpo_config = DPOConfig(**only_supported(DPOConfig, config_kwargs))

    trainer_kwargs = dict(model=model, ref_model=None, args=dpo_config, train_dataset=train_ds,
                          eval_dataset=val_ds, peft_config=lora)
    params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    start = time.time()
    initial = trainer.evaluate()
    initial["step"] = 0
    print("step 0 eval:", {k: round(v, 4) for k, v in initial.items() if isinstance(v, float)}, flush=True)
    trainer.train()
    final = trainer.evaluate()
    minutes = (time.time() - start) / 60

    history = trainer.state.log_history  # includes the step-0 and final evaluations
    metrics = to_metrics(history)
    (out / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    (out / "log_history.json").write_text(json.dumps(history, indent=1))

    trainer.model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out / "merged")

    run_config = {**vars(args), **versions, "num_train": len(train_ds), "num_val": len(val_ds),
                  "trainable_params": trainable, "total_params": total,
                  "effective_batch_size": args.batch_size * args.grad_accum,
                  "planned_steps": planned_steps, "warmup_steps": warmup_steps,
                  "note": "TRL 1.x has no max_prompt_length; only max_length truncation applies",
                  "precision": "fp16 autocast + fp32 weights" if use_fp16 else "fp32",
                  "reference": "base model with LoRA adapter disabled (TRL PEFT default)",
                  "dpo_config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                 for k, v in dpo_config.to_dict().items()}}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    summary = {"initial_eval": initial, "final_eval": final, "global_steps": trainer.state.global_step,
               "minutes": minutes}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("final eval:", {k: round(v, 4) for k, v in final.items() if isinstance(v, float)})
    print(f"done in {minutes:.1f} min; adapter -> {out / 'adapter'}, merged model -> {out / 'merged'}")


if __name__ == "__main__":
    main()
