"""Build a blinded pairwise-comparison JSONL file from two generation files.

    python scripts/build_pairs.py \
        --target outputs/generations/dpo_uf_alpaca_eval_rep1.1.json \
        --baseline outputs/generations/sft_alpaca_eval_rep1.1.json \
        --output outputs/eval/dpo_uf_vs_sft_alpaca_pairs.jsonl

For every prompt id present in both files (in the eval-set order of the target
file), the two responses are placed in positions A/B by a seeded coin flip. The
judge only ever sees ``instruction``, ``response_a``, ``response_b`` (and, for the
HH eval set, the item's ``expected_behavior`` rubric note); which model produced
which position is stored under ``hidden`` and is only used after judging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, help="generations of the model being evaluated (DPO)")
    p.add_argument("--baseline", required=True, help="generations of the baseline (SFT)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    target, baseline = load(args.target), load(args.baseline)
    base_by_id = {row["id"]: row for row in baseline}
    target_name, baseline_name = target[0]["generator"], baseline[0]["generator"]
    if target_name == baseline_name:
        raise SystemExit(f"both files have generator '{target_name}'")
    settings = {}
    for side, path in (("target", args.target), ("baseline", args.baseline)):
        sp = Path(path).with_suffix(".settings.json")
        if sp.exists():
            s = json.loads(sp.read_text())
            settings[side] = {k: s.get(k) for k in ("decoding", "max_new_tokens", "repetition_penalty",
                                                    "no_repeat_ngram_size", "temperature", "top_p")}
    if len(settings) == 2 and settings["target"] != settings["baseline"]:
        raise SystemExit(f"decoding settings differ between the two files: {settings}")

    rng = random.Random(args.seed)
    rows, missing = [], 0
    for t in target:
        b = base_by_id.get(t["id"])
        if b is None:
            missing += 1
            continue
        if t["instruction"] != b["instruction"]:
            raise SystemExit(f"instruction mismatch for id {t['id']}")
        target_is_a = rng.random() < 0.5
        a, bb = (t, b) if target_is_a else (b, t)
        row = {
            "pair_id": t["id"],
            "dataset": t.get("dataset", ""),
            "category": t.get("category"),
            "instruction": t["instruction"],
            "expected_behavior": t.get("expected_behavior"),
            "response_a": a["output"],
            "response_b": bb["output"],
            "hidden": {"model_a": a["generator"], "model_b": bb["generator"],
                       "target": target_name, "baseline": baseline_name},
        }
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(text, encoding="utf-8")
    positions = Counter("A" if r["hidden"]["model_a"] == target_name else "B" for r in rows)
    print(f"wrote {len(rows)} pairs to {out} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]})")
    print(f"target={target_name} baseline={baseline_name} | target position counts {dict(positions)} "
          f"| missing in baseline: {missing} | decoding: {settings.get('target')}")
    identical = sum(r["response_a"].strip() == r["response_b"].strip() for r in rows)
    print(f"pairs with identical responses: {identical}")


if __name__ == "__main__":
    main()
