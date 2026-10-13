"""Extra checks beyond the public smoke tests: truncation, padding, row order, math."""

from __future__ import annotations

import math

import torch

import submission_adapter as adapter
from alignment.losses import dpo_reward_metrics
from alignment.logprobs import batch_logprobs
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    """Like many causal-LM tokenizers: has EOS but no pad token id."""

    pad_token_id = None


def _manual_logp(model, tokenizer, token_ids):
    """Toy model predicts the same distribution everywhere: log_softmax(bias)."""
    log_probs = torch.log_softmax(model.logit_bias.detach().float(), dim=-1)
    return sum(log_probs[t].item() for t in token_ids)


def test_eos_is_dropped_when_truncation_cuts_the_response():
    tokenizer = ToyTokenizer()
    # "Q: " = 3 tokens, "hello" + EOS = 6 tokens; max_length 6 keeps only "hel".
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["hello"], max_length=6)
    labels = batch["labels"][0]
    assert batch["input_ids"].shape == (1, 6)
    assert labels.ne(-100).sum().item() == 3
    assert tokenizer.eos_token_id not in labels.tolist()


def test_prompt_longer_than_max_length_gives_no_response_tokens_and_finite_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tokenizer, ["abcdefgh"], ["xyz"], max_length=4)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tokenizer, ["abcdefgh"], ["xyz"], 4)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tokenizer, ["abcdefgh"], ["xyz"], 4)
    assert torch.isfinite(loss)


def test_missing_pad_token_falls_back_to_eos_and_padding_is_masked():
    tokenizer = NoPadTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["A: ", "Longer prompt: "], ["x", "y"], 32)
    short_len = len(tokenizer.encode("A: ")) + 2  # "x" + EOS
    assert batch["attention_mask"][0, short_len:].eq(0).all()
    assert batch["labels"][0, short_len:].eq(-100).all()
    assert batch["input_ids"][0, short_len:].eq(tokenizer.eos_token_id).all()


def test_row_order_is_preserved_with_mixed_lengths():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.9)
    prompts = ["P: ", "Prompt two: ", "3: "]
    responses = ["abc", "z", "hello there"]
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, 64)
    for row, response in enumerate(responses):
        ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
        expected = _manual_logp(model, tokenizer, ids)
        assert math.isclose(logps[row].item(), expected, rel_tol=1e-5, abs_tol=1e-5)


def test_dpo_loss_matches_formula_for_unequal_logps():
    pc = torch.tensor([-3.0, -1.0])
    pr = torch.tensor([-4.0, -2.5])
    rc = torch.tensor([-3.5, -1.0])
    rr = torch.tensor([-3.5, -2.0])
    beta = 0.1
    logits = beta * ((pc - pr) - (rc - rr))
    expected = -torch.log(torch.sigmoid(logits)).mean()
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, expected)


def test_one_dpo_step_makes_policy_prefer_chosen():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    args = (tokenizer, ["Pick: "], ["a"], ["b"])

    batch = adapter.build_dpo_batch(*args, max_length=16)
    with torch.no_grad():
        ref_c, _ = batch_logprobs(reference, batch["chosen"])
        ref_r, _ = batch_logprobs(reference, batch["rejected"])
        pol_c, _ = batch_logprobs(policy, batch["chosen"])
        pol_r, _ = batch_logprobs(policy, batch["rejected"])
    before = dpo_reward_metrics(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert before["reward_accuracy"] == 0.0  # identical models: all rewards tied
    assert before["reward_margin"] == 0.0

    loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.1, max_length=16)
    loss.backward()
    with torch.no_grad():
        policy.logit_bias -= 1.0 * policy.logit_bias.grad

    with torch.no_grad():
        pol_c, _ = batch_logprobs(policy, batch["chosen"])
        pol_r, _ = batch_logprobs(policy, batch["rejected"])
    after = dpo_reward_metrics(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert after["reward_accuracy"] == 1.0
    assert after["reward_margin"] > 0.0
