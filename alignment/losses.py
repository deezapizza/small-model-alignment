"""SFT and DPO objectives.

SFT reduction (used everywhere in this repo): the mean negative log-likelihood
over ALL unmasked response tokens in the batch, i.e.
    loss = -(sum of response-token log-probs) / (number of response tokens).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batching import build_dpo_batch
from .logprobs import batch_logprobs


def sft_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Token-mean response-only negative log-likelihood (scalar)."""
    logps, token_counts = batch_logprobs(model, batch)
    return -logps.sum() / token_counts.sum().clamp_min(1)


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Batch-mean DPO loss: mean of -log sigmoid(beta * (Δ_policy - Δ_reference)).

    Δ_policy    = log π(y_w|x) - log π(y_l|x)
    Δ_reference = log π_ref(y_w|x) - log π_ref(y_l|x)
    ``F.logsigmoid`` is used instead of log(sigmoid(.)) to avoid log(0).
    """
    device = policy_chosen_logps.device
    policy_gap = policy_chosen_logps - policy_rejected_logps.to(device)
    reference_gap = reference_chosen_logps.to(device) - reference_rejected_logps.to(device)
    logits = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()


@torch.no_grad()
def dpo_reward_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> dict[str, float]:
    """Implicit rewards r = beta * (log π - log π_ref) and the manual's metrics.

    reward_accuracy uses the STRICT comparison r_w > r_l, so it is exactly 0 when
    the policy still equals the reference (all rewards tied at 0).
    """
    device = policy_chosen_logps.device
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps.to(device))
    rejected_rewards = beta * (policy_rejected_logps.to(device) - reference_rejected_logps.to(device))
    return {
        "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


def dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta: float,
    max_length: int,
    max_prompt_length: int | None = None,
    return_metrics: bool = False,
):
    """Full DPO step: tokenize, score with policy (grad) and reference (no grad), compute loss.

    The policy and reference may live on different devices (e.g. cuda:0 / cuda:1);
    reference log-probs are moved to the policy's device before combining.
    """
    batch = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_length
    )
    policy_chosen, _ = batch_logprobs(policy_model, batch["chosen"])
    policy_rejected, _ = batch_logprobs(policy_model, batch["rejected"])
    with torch.no_grad():
        reference_chosen, _ = batch_logprobs(reference_model, batch["chosen"])
        reference_rejected, _ = batch_logprobs(reference_model, batch["rejected"])

    loss = dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    if not return_metrics:
        return loss
    metrics = dpo_reward_metrics(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    return loss, metrics
