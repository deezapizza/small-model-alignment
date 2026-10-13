"""Create the fixed AlpacaEval evaluation subset (default: 300 prompts, seed 42).

    python scripts/make_alpaca_eval_subset.py

Downloads ``alpaca_eval.json`` (805 prompts) from the ``tatsu-lab/alpaca_eval``
dataset repo, samples ``n`` distinct prompts with ``random.Random(seed)``, keeps
them in their original dataset order, and writes
``eval_sets/alpaca_eval_<n>_seed<seed>.json``. Commit that file: every model
(base, SFT, DPO) must be generated on exactly the same prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="default: eval_sets/alpaca_eval_<n>_seed<seed>.json")
    args = parser.parse_args()

    source_path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(source_path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"loaded {len(rows)} AlpacaEval prompts from {source_path}")
    if args.n > len(rows):
        raise SystemExit(f"--n {args.n} is larger than the dataset ({len(rows)} prompts)")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(rows)), args.n))
    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{position:04d}",
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "source": rows[index].get("dataset"),
            "original_index": index,
        }
        for position, index in enumerate(indices)
    ]

    output = Path(args.output or f"eval_sets/alpaca_eval_{args.n}_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(subset, indent=2, ensure_ascii=False) + "\n"
    output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    print(f"wrote {len(subset)} prompts to {output} (sha256 {digest})")
    print("first ids:", [row["id"] for row in subset[:3]])


if __name__ == "__main__":
    main()
