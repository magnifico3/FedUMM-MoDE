#!/usr/bin/env python3

"""Plot per-round FL training metrics from NVFlare simulator logs."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TRAIN_LINE_RE = re.compile(
    r"\[(?P<site>[^\]]+)\]\s+train\s+round=(?P<round>\d+)\s+loss=(?P<loss>[0-9.]+)\s+acc=(?P<acc>[0-9.]+)"
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot FL loss/accuracy curves from simulator logs.")
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace_simulator",
        help="Path to NVFlare simulator workspace.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for generated plots. Defaults to <workspace>/plots.",
    )
    return parser.parse_args()


def load_histories(workspace_dir: Path) -> dict[str, list[dict]]:
    histories = defaultdict(list)
    for log_path in sorted(workspace_dir.glob("site-*/log.json")):
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = record.get("message", "")
                match = TRAIN_LINE_RE.search(message)
                if not match:
                    continue
                histories[match.group("site")].append(
                    {
                        "round": int(match.group("round")),
                        "loss": float(match.group("loss")),
                        "acc": float(match.group("acc")),
                    }
                )
    for site, rows in histories.items():
        rows.sort(key=lambda item: item["round"])
    return dict(histories)


def compute_mean_series(histories: dict[str, list[dict]], metric: str) -> list[tuple[int, float]]:
    grouped = defaultdict(list)
    for rows in histories.values():
        for row in rows:
            grouped[row["round"]].append(row[metric])
    return sorted((round_id, sum(values) / len(values)) for round_id, values in grouped.items())


def plot_metric(histories: dict[str, list[dict]], metric: str, ylabel: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib` "
            "or add it to your conda environment."
        ) from e

    plt.figure(figsize=(10, 6))
    for site, rows in sorted(histories.items()):
        rounds = [row["round"] for row in rows]
        values = [row[metric] for row in rows]
        plt.plot(rounds, values, marker="o", linewidth=1.8, label=site, alpha=0.85)

    mean_series = compute_mean_series(histories, metric)
    if mean_series:
        plt.plot(
            [round_id for round_id, _ in mean_series],
            [value for _, value in mean_series],
            marker="s",
            linestyle="--",
            linewidth=2.4,
            color="black",
            label="mean",
        )

    plt.title(f"Federated Training {ylabel} by Round")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.xticks(sorted({row["round"] for rows in histories.values() for row in rows}))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary_csv(histories: dict[str, list[dict]], output_path: Path) -> None:
    lines = ["site,round,loss,acc"]
    for site, rows in sorted(histories.items()):
        for row in rows:
            lines.append(f"{site},{row['round']},{row['loss']:.6f},{row['acc']:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    workspace_dir = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else workspace_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(workspace_dir)
    if not histories:
        raise SystemExit(f"No train metrics found under {workspace_dir}.")

    plot_metric(histories, metric="loss", ylabel="Loss", output_path=output_dir / "train_loss.png")
    plot_metric(histories, metric="acc", ylabel="Accuracy", output_path=output_dir / "train_accuracy.png")
    write_summary_csv(histories, output_dir / "metrics_summary.csv")

    print(f"Saved plots to: {output_dir}")
    print(f"  - {output_dir / 'train_loss.png'}")
    print(f"  - {output_dir / 'train_accuracy.png'}")
    print(f"  - {output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
