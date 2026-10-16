"""Combine original-order and swapped-order judgments into position-debiased results.

    python scripts/combine_swapped.py \
        outputs/eval/dpo_hh_vs_sft_hh_judged.jsonl \
        outputs/eval/dpo_hh_vs_sft_hh_swapped_judged.jsonl \
        --output outputs/eval/dpo_hh_vs_sft_hh_combined.json

Per pair (decisions are from the target model's point of view):
  * same decision in both orders           -> that decision (win / loss / tie)
  * win in one order, tie in the other     -> win (likewise loss)
  * win in one order, loss in the other    -> tie  ("position-inconsistent")
  * missing / unknown in either order      -> unknown
Also reports a two-sided sign test and a 95% Wilson interval for the
ties-excluded win rate, for both the single-order and the combined results.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    return {r["pair_id"]: r for r in rows}


def combine(d1: str | None, d2: str | None) -> tuple[str, bool]:
    if d1 is None or d2 is None or "unknown" in (d1, d2):
        return "unknown", False
    if d1 == d2:
        return d1, False
    pair = {d1, d2}
    if pair == {"win", "loss"}:
        return "tie", True
    return ("win" if "win" in pair else "loss"), False


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def stats(counter: Counter) -> dict:
    w, l, t, u = counter["win"], counter["loss"], counter["tie"], counter["unknown"]
    return {"wins": w, "losses": l, "ties": t, "unknown": u,
            "win_rate_excluding_ties": round(w / (w + l), 4) if w + l else None,
            "win_rate_ties_half": round((w + 0.5 * t) / (w + l + t), 4) if w + l + t else None,
            "wilson_95ci_excluding_ties": wilson(w, w + l),
            "sign_test_p_two_sided": round(sign_test_p(w, l), 4)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("original")
    p.add_argument("swapped")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    first, second = load(args.original), load(args.swapped)
    ids = list(first) + [i for i in second if i not in first]

    combined, inconsistent = {}, 0
    per_category: dict[str, Counter] = {}
    agreement = Counter()
    for pair_id in ids:
        a, b = first.get(pair_id), second.get(pair_id)
        decision, flip = combine(a and a["decision"], b and b["decision"])
        inconsistent += flip
        if a and b:
            agreement["same" if a["decision"] == b["decision"] else "different"] += 1
        category = (a or b).get("category") or "all"
        per_category.setdefault(category, Counter())[decision] += 1
        combined[pair_id] = {"pair_id": pair_id, "category": category, "decision": decision,
                             "original": a and a["decision"], "swapped": b and b["decision"],
                             "position_inconsistent": flip}

    def position_share(rows: dict) -> float | None:
        c = Counter(r["winner"] for r in rows.values())
        return round(c["A"] / (c["A"] + c["B"]), 4) if c["A"] + c["B"] else None

    summary = {
        "original": args.original, "swapped": args.swapped, "num_pairs": len(ids),
        "single_order_original": stats(Counter(r["decision"] for r in first.values())),
        "single_order_swapped": stats(Counter(r["decision"] for r in second.values())),
        "combined": stats(Counter(r["decision"] for r in combined.values())),
        "position_inconsistent_win_loss_flips": inconsistent,
        "decision_agreement_between_orders": dict(agreement),
        "judge_picked_A_share": {"original": position_share(first), "swapped": position_share(second)},
        "combined_by_category": {k: dict(v) for k, v in sorted(per_category.items())},
    }
    out = Path(args.output)
    out.write_text(json.dumps({"summary": summary, "pairs": list(combined.values())}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
