"""Plot DPO curves (loss, reward accuracy, reward margin) from <run_dir>/metrics.jsonl.

    python scripts/plot_dpo.py outputs/dpo_hh       # writes outputs/dpo_hh/dpo_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ema(values, alpha=0.9):
    out, running = [], None
    for v in values:
        running = v if running is None else alpha * running + (1 - alpha) * v
        out.append(running)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run = Path(args.run_dir)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    train = [r for r in records if r["type"] == "train"]
    val = [r for r in records if r["type"] == "val"]

    panels = [("loss", "train_loss", "val_loss", "DPO loss"),
              ("reward accuracy", "train_reward_accuracy", "val_reward_accuracy", "fraction r_w > r_l"),
              ("reward margin", "train_reward_margin", "val_reward_margin", "mean r_w − r_l")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (name, tkey, vkey, ylabel) in zip(axes, panels):
        steps = [r["step"] for r in train]
        values = [r[tkey] for r in train]
        ax.plot(steps, values, color="#2a78d6", alpha=0.25, linewidth=1)
        ax.plot(steps, ema(values), color="#2a78d6", linewidth=2, label="train (EMA)")
        ax.plot([r["step"] for r in val], [r[vkey] for r in val], color="#eb6834", linewidth=2,
                marker="o", markersize=3, label="validation")
        ax.set_title(name, loc="left", fontsize=10, fontweight="semibold")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(args.title or run.name, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) if args.out else run / "dpo_curves.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
