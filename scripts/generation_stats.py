"""Length and repetition statistics for generation files (used to pick decoding settings).

    python scripts/generation_stats.py outputs/generations/*.json
    python scripts/generation_stats.py outputs/generations/*.json --tokenizer Qwen/Qwen2.5-0.5B --csv outputs/eval/generation_stats.csv

Per file: number of responses, empty responses, mean/median words, and two
degeneration signals:
  * loop rate: share of responses in which some 6-word sequence occurs >= 3 times;
  * distinct-2 (unique bigrams / bigrams), averaged over responses.
With --tokenizer, also mean tokens and the share of responses that hit
max_new_tokens (from the .settings.json), i.e. never stopped on their own.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def has_loop(words: list[str], n: int = 6, times: int = 3) -> bool:
    if len(words) < n * times:
        return False
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(counts.values()) >= times


def distinct2(words: list[str]) -> float:
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for path in args.files:
        if path.endswith(".settings.json"):
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        outputs = [d["output"] for d in data]
        words = [o.split() for o in outputs]
        row = {
            "file": Path(path).name, "n": len(outputs),
            "empty": sum(not o.strip() for o in outputs),
            "mean_words": round(statistics.mean(len(w) for w in words), 1),
            "median_words": statistics.median(len(w) for w in words),
            "loop_rate": round(sum(has_loop(w) for w in words) / len(words), 3),
            "distinct2": round(statistics.mean(distinct2(w) for w in words), 3),
        }
        if tok is not None:
            settings_path = Path(path).with_suffix(".settings.json")
            max_new = json.loads(settings_path.read_text()).get("max_new_tokens", 512) if settings_path.exists() else 512
            lengths = [len(tok.encode(o, add_special_tokens=False)) for o in outputs]
            row["mean_tokens"] = round(statistics.mean(lengths), 1)
            row["hit_max_tokens"] = round(sum(n >= max_new - 2 for n in lengths) / len(lengths), 3)
        rows.append(row)

    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
