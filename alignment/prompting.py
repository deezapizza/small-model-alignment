"""The single prompt format used for SFT, DPO and every generation run.

We use Qwen's ChatML layout written out explicitly (instead of calling
``tokenizer.apply_chat_template``) so the exact string is fixed in code and can
be quoted in the report. The base model has never been trained on this format;
SFT teaches it, and every model is evaluated with the same prompt so the
comparison is fair.

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<EOS>

The response is followed by the tokenizer's EOS token (``<|endoftext|>`` for
Qwen2.5), appended by ``build_lm_batch``.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."
IM_END = "<|im_end|>"


def format_prompt(instruction: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Wrap a single-turn instruction in the ChatML prompt, ending at the assistant turn."""
    return (
        f"<|im_start|>system\n{system_prompt}{IM_END}\n"
        f"<|im_start|>user\n{instruction.strip()}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that end generation: EOS, plus ``<|im_end|>`` if the tokenizer has it."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    if IM_END in vocab and int(vocab[IM_END]) not in ids:
        ids.append(int(vocab[IM_END]))
    return ids
