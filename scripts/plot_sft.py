"""Plot SFT training and validation loss from <run_dir>/metrics.jsonl.

    python scripts/plot_sft.py outputs/sft          # writes outputs/sft/sft_loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run = Path(args.run_dir)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    train = [r for r in records if r["type"] == "train"]
    val = [r for r in records if r["type"] == "val"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if train:
        steps = [r["step"] for r in train]
        losses = [r["train_loss"] for r in train]
        smooth, running = [], None
        for value in losses:
            running = value if running is None else 0.9 * running + 0.1 * value
            smooth.append(running)
        ax.plot(steps, losses, color="#2a78d6", alpha=0.25, linewidth=1)
        ax.plot(steps, smooth, color="#2a78d6", linewidth=2, label="train (EMA 0.9 of logged steps)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], color="#eb6834",
                linewidth=2, marker="o", markersize=4, label="validation (1,000 held-out examples)")
        best = min(val, key=lambda r: r["val_loss"])
        ax.annotate(f"best {best['val_loss']:.3f} @ step {best['step']}", (best["step"], best["val_loss"]),
                    xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_title("SFT: response-only token-mean NLL", loc="left", fontsize=11, fontweight="semibold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (nats / response token)")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(args.out) if args.out else run / "sft_loss.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
