"""Turn raw preference datasets into (instruction, chosen, rejected) triples.

Pure Python (no torch), so the parsing rules are easy to test and to quote in
the report.

* Anthropic/hh-rlhf: each row is two full transcripts
  ``"\\n\\nHuman: ... \\n\\nAssistant: ..."``. We keep only SINGLE-TURN rows
  (exactly one Human and one Assistant turn in both transcripts, with the same
  Human turn), so the instruction can be wrapped in our ChatML prompt just like
  at evaluation time.
* HuggingFaceH4/ultrafeedback_binarized: ``prompt`` plus ``chosen`` / ``rejected``
  message lists; the response is the final assistant message. Rows with tied
  scores are dropped.
Pairs whose two responses are identical (after stripping) or empty are dropped.
"""

from __future__ import annotations

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"


def _split_single_turn(transcript: str) -> tuple[str, str] | None:
    text = transcript if transcript.startswith("\n\n") else "\n\n" + transcript.lstrip()
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    human_at, assistant_at = text.index(HUMAN), text.index(ASSISTANT)
    if human_at != 0 or assistant_at < human_at:
        return None
    instruction = text[len(HUMAN) : assistant_at].strip()
    response = text[assistant_at + len(ASSISTANT) :].strip()
    return instruction, response


def parse_hh(chosen: str, rejected: str) -> tuple[str, str, str] | None:
    """(instruction, chosen_response, rejected_response) or None if not a usable single-turn pair."""
    a, b = _split_single_turn(chosen), _split_single_turn(rejected)
    if a is None or b is None or a[0] != b[0]:
        return None
    instruction, chosen_response, rejected_response = a[0], a[1], b[1]
    if not instruction or not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:
        return None
    return instruction, chosen_response, rejected_response


def _last_assistant(messages) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "assistant":
            return (message.get("content") or "").strip()
    return ""


def parse_ultrafeedback(row: dict) -> tuple[str, str, str] | None:
    instruction = (row.get("prompt") or "").strip()
    chosen_response = _last_assistant(row.get("chosen"))
    rejected_response = _last_assistant(row.get("rejected"))
    if not instruction or not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:
        return None
    score_c, score_r = row.get("score_chosen"), row.get("score_rejected")
    if score_c is not None and score_r is not None and score_c <= score_r:
        return None
    return instruction, chosen_response, rejected_response


DATASETS = {
    "hh": {"hf_id": "Anthropic/hh-rlhf", "train_split": "train", "val_split": "test"},
    "ultrafeedback": {"hf_id": "HuggingFaceH4/ultrafeedback_binarized",
                      "train_split": "train_prefs", "val_split": "test_prefs"},
}


def parse_row(name: str, row: dict) -> tuple[str, str, str] | None:
    if name == "hh":
        return parse_hh(row["chosen"], row["rejected"])
    if name == "ultrafeedback":
        return parse_ultrafeedback(row)
    raise ValueError(f"unknown dataset {name!r}")
