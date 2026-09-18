"""Generate training curves and a final comparison figure for the report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "report" / "figures"


def load_history(path: Path) -> list[dict[str, float | int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["history"]


def plot_training_curves(
    history: list[dict[str, float | int]],
    title: str,
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_top1_accuracy"] for row in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_top1_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.65, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(output_path: Path) -> None:
    labels = ["Baseline\nbest val", "Label Smoothing\nbest val", "Final\ntest"]
    top1 = [0.9247, 0.9319, 0.9266]
    macro_f1 = [0.9244, 0.9315, 0.9258]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(positions, top1, color=["#4472C4", "#ED7D31", "#70AD47"])
    axes[0].set_title("Top-1 Accuracy")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0.88, 0.95)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(top1):
        axes[0].text(index, value + 0.001, f"{value:.4f}", ha="center")

    axes[1].bar(positions, macro_f1, color=["#4472C4", "#ED7D31", "#70AD47"])
    axes[1].set_title("Macro-F1")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0.88, 0.95)
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(macro_f1):
        axes[1].text(index, value + 0.001, f"{value:.4f}", ha="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_history = load_history(
        PROJECT_ROOT / "artifacts" / "runs" / "baseline" / "metrics.json"
    )
    smoothing_history = load_history(
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "label_smoothing_0.1"
        / "metrics.json"
    )

    plot_training_curves(
        baseline_history,
        "Baseline ResNet-18",
        OUTPUT_DIR / "baseline_training_curves.png",
    )
    plot_training_curves(
        smoothing_history,
        "ResNet-18 with Label Smoothing 0.1",
        OUTPUT_DIR / "label_smoothing_training_curves.png",
    )
    plot_comparison(OUTPUT_DIR / "final_comparison.png")

    print("Generated report figures:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
