"""Summed log-probability of the response tokens of each row (prompt and padding excluded)."""

from __future__ import annotations

import torch

from .batching import IGNORE_INDEX


def model_device(model) -> torch.device:
    """Device of the model's first parameter (CPU if it has none)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(summed response log-probs [B], number of scored tokens [B])``.

    The logits at position t predict the token at position t + 1, so we compare
    ``logits[:, :-1]`` with ``labels[:, 1:]``. Positions whose label is -100
    (prompt and padding) contribute exactly 0. Log-softmax is done in float32
    for numerical stability, even if the model runs in fp16.
    """
    device = model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()  # [B, T-1, V]
    targets = labels[:, 1:]  # [B, T-1]
    mask = targets.ne(IGNORE_INDEX)

    safe_targets = targets.masked_fill(~mask, 0)  # any valid id; masked out below
    token_logps = torch.log_softmax(logits, dim=-1)
    token_logps = token_logps.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~mask, 0.0)

    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def batch_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """``sequence_logprobs`` for a batch produced by ``build_lm_batch``."""
    return sequence_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["labels"]
    )
