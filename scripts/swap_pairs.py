"""Make a position-swapped copy of a pairs file (Response A <-> Response B).

    python scripts/swap_pairs.py outputs/eval/dpo_hh_vs_sft_hh_pairs.jsonl
    # -> outputs/eval/dpo_hh_vs_sft_hh_pairs_swapped.jsonl

Judging both files and combining them with scripts/combine_swapped.py controls for
the judge's position bias: a model only gets a win if it wins in BOTH orders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_swapped.jsonl")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["response_a"], row["response_b"] = row["response_b"], row["response_a"]
        hidden = row["hidden"]
        hidden["model_a"], hidden["model_b"] = hidden["model_b"], hidden["model_a"]
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} swapped pairs to {dst}")


if __name__ == "__main__":
    main()
