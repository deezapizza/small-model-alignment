
from __future__ import annotations

from alignment.batching import build_dpo_batch as _build_dpo_batch
from alignment.batching import build_lm_batch as _build_lm_batch
from alignment.logprobs import batch_logprobs as _batch_logprobs
from alignment.losses import dpo_loss as _dpo_loss
from alignment.losses import dpo_loss_from_logps as _dpo_loss_from_logps
from alignment.losses import sft_loss as _sft_loss


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return _build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = _build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = _batch_logprobs(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = _build_lm_batch(tokenizer, prompts, responses, max_length)
    return _sft_loss(model, batch)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return _build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):  return _dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )

def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    return _dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
