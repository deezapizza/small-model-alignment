"""Generate one response per prompt of a fixed evaluation set and save them as JSON.

    # quick sanity check on 3 prompts
    python scripts/generate.py --model Qwen/Qwen2.5-0.5B \
        --eval_set eval_sets/alpaca_eval_300_seed42.json \
        --output outputs/generations/base_alpaca_eval.json \
        --generator_name qwen2.5-0.5b-base --limit 3

    # full run (resumes automatically if the output file already has some ids)
    python scripts/generate.py --model Qwen/Qwen2.5-0.5B \
        --eval_set eval_sets/alpaca_eval_300_seed42.json \
        --output outputs/generations/base_alpaca_eval.json \
        --generator_name qwen2.5-0.5b-base

Every prompt is wrapped with ``alignment.prompting.format_prompt`` (the same
ChatML format used for SFT and DPO). Decoding is greedy by default. The exact
decoding settings are written next to the output as ``<output>.settings.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.prompting import SYSTEM_PROMPT, format_prompt, stop_token_ids  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate responses for a fixed evaluation set.")
    p.add_argument("--model", required=True, help="HF model id or local checkpoint directory")
    p.add_argument("--eval_set", required=True, help="JSON list with at least 'id' and 'instruction'")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--generator_name", required=True, help="e.g. qwen2.5-0.5b-base, sft, dpo-uf")
    p.add_argument("--dataset_name", default=None, help="overrides each item's 'dataset' field")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 = off")
    p.add_argument("--no_repeat_ngram_size", type=int, default=0, help="0 = off")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    p.add_argument("--prompt_format", choices=["chatml", "tokenizer"], default="chatml",
                   help="chatml = alignment.prompting.format_prompt (Qwen SFT/DPO models); "
                        "tokenizer = the model's own chat template (SmolLM2 persona models)")
    return p.parse_args()


def write_json_atomic(path: Path, obj) -> None:
    """Write to a temp file then rename, so a disconnect never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    with open(args.eval_set, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit is not None:
        items = items[: args.limit]
    for item in items:
        if "id" not in item or "instruction" not in item:
            raise SystemExit(f"every eval item needs 'id' and 'instruction'; got keys {sorted(item)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if output.exists():
        done = {row["id"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
        print(f"resuming: {len(done)} generations already in {output}")
    todo = [item for item in items if item["id"] not in done]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype])
    model.to(device).eval()

    sampling = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=sampling,
        temperature=args.temperature if sampling else None,
        top_p=args.top_p if sampling else None,
        top_k=None,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    settings = {
        "model": args.model,
        "generator_name": args.generator_name,
        "eval_set": args.eval_set,
        "num_prompts": len(items),
        "prompt_format": ("ChatML (alignment.prompting.format_prompt)" if args.prompt_format == "chatml"
                          else "tokenizer.apply_chat_template (user turn only, add_generation_prompt=True)"),
        "system_prompt": SYSTEM_PROMPT if args.prompt_format == "chatml" else "tokenizer template default",
        "decoding": "sampling" if sampling else "greedy",
        **{k: v for k, v in gen_kwargs.items() if k not in ("pad_token_id",)},
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": device,
        "torch": torch.__version__,
    }
    write_json_atomic(output.with_suffix(".settings.json"), settings)
    print(json.dumps(settings, indent=2))

    if not todo:
        print("nothing to generate")
        return
    start_time = time.time()
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        if args.prompt_format == "chatml":
            prompts = [format_prompt(item["instruction"]) for item in chunk]
        else:
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": item["instruction"].strip()}],
                                                     tokenize=False, add_generation_prompt=True)
                       for item in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, **gen_kwargs)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for item, text in zip(chunk, texts):
            record = {
                "id": item["id"],
                "instruction": item["instruction"],
                "output": text.strip(),
                "generator": args.generator_name,
                "dataset": args.dataset_name or item.get("dataset", ""),
            }
            for key, value in item.items():  # keep extra metadata (category, source, ...)
                record.setdefault(key, value)
            done[item["id"]] = record

        ordered = [done[item["id"]] for item in items if item["id"] in done]
        write_json_atomic(output, ordered)
        elapsed = time.time() - start_time
        finished = min(start + args.batch_size, len(todo))
        print(f"{finished}/{len(todo)} generated ({elapsed / 60:.1f} min) -> {output}", flush=True)

    print(f"done: {len(done)} generations in {output}")


if __name__ == "__main__":
    main()
