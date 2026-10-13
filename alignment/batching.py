
from __future__ import annotations

from collections.abc import Sequence

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_token_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    if len(prompts) != len(responses):
        raise ValueError(
            f"got {len(prompts)} prompts but {len(responses)} responses; they must match"
        )
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = _pad_token_id(tokenizer)

    rows: list[tuple[list[int], list[int]]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :]

        response_ids = _encode(tokenizer, response)
        if eos_id is not None:
            response_ids = response_ids + [int(eos_id)]

        input_ids = (prompt_ids + response_ids)[:max_length]
        n_prompt = min(len(prompt_ids), len(input_ids))
        labels = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]
        rows.append((input_ids, labels))

    batch_size = len(rows)
    width = max([len(ids) for ids, _ in rows], default=0)
    width = max(width, 1) 

    input_ids_t = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask_t = torch.zeros((batch_size, width), dtype=torch.long)
    labels_t = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labels) in enumerate(rows):
        length = len(ids)
        if length == 0:
            continue
        input_ids_t[row, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask_t[row, :length] = 1
        labels_t[row, :length] = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_ids_t, "attention_mask": attention_mask_t, "labels": labels_t}


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Two LM batches with the SAME prompt formatting: one chosen, one rejected."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError(
            "prompts, chosen_responses and rejected_responses must have the same length, got "
            f"{len(prompts)}, {len(chosen_responses)}, {len(rejected_responses)}"
        )
    return {
        "chosen": build_lm_batch(
            tokenizer, prompts, chosen_responses, max_length, max_prompt_length
        ),
        "rejected": build_lm_batch(
            tokenizer, prompts, rejected_responses, max_length, max_prompt_length
        ),
    }
