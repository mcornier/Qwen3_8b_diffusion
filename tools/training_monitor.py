import argparse
import json
import re
from typing import List, Dict

import matplotlib.pyplot as plt


def parse_log_file(path: str) -> Dict[str, List[float]]:
    """Parse a training log file and return epochs, losses and learning rates."""
    epoch_pattern = re.compile(r"epoch\s*[:=]\s*(\d+)", re.IGNORECASE)
    loss_pattern = re.compile(r"loss\s*[:=]\s*([0-9\.eE+-]+)", re.IGNORECASE)
    lr_pattern = re.compile(r"lr\s*[:=]\s*([0-9\.eE+-]+)", re.IGNORECASE)

    epochs, losses, lrs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            loss_match = loss_pattern.search(line)
            lr_match = lr_pattern.search(line)
            if epoch_match and loss_match:
                epoch = int(epoch_match.group(1))
                loss = float(loss_match.group(1))
                lr = float(lr_match.group(1)) if lr_match else None

                epochs.append(epoch)
                losses.append(loss)
                lrs.append(lr)
    return {"epochs": epochs, "losses": losses, "lrs": lrs}


def plot_training(data: Dict[str, List[float]], output_prefix: str) -> None:
    """Plot training curves and save figure."""
    epochs = data["epochs"]
    losses = data["losses"]
    lrs = data["lrs"]

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, losses, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    if any(lrs):
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Learning Rate", color=color)
        ax2.plot(epochs, lrs, color=color, linestyle="--", marker="x")
        ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Progress")
    fig.savefig(f"{output_prefix}.png")


def save_json(data: Dict[str, List[float]], output_prefix: str) -> None:
    """Save training data to JSON file."""
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training supervision tool")
    parser.add_argument("log_path", help="Path to training log file")
    parser.add_argument("output", help="Output prefix for saved files")
    args = parser.parse_args()

    data = parse_log_file(args.log_path)
    save_json(data, args.output)
    plot_training(data, args.output)


if __name__ == "__main__":
    main()
