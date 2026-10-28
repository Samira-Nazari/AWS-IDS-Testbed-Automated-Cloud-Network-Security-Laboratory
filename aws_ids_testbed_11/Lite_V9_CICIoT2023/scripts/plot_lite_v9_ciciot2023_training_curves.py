#!/usr/bin/env python3
"""
Plot Lite_V9_CICIoT2023 training/evaluation diagnostics.

Default inputs:
  outputs/logs/lite_v9_ciciot2023_step_train_16913_20260616_010550.txt
  outputs/results/predictions.csv
  outputs/data/label_classes.pkl
  outputs/data/step_2_Processed_Data.pkl

Default outputs:
  outputs/results/lite_v9_ciciot2023_training_curves.png
  outputs/results/lite_v9_ciciot2023_label_distribution_analysis.png
  outputs/results/lite_v9_ciciot2023_probability_heatmap_by_true_class.png
  outputs/results/lite_v9_ciciot2023_predicted_class_1_probability_by_true_class.png
  outputs/results/lite_v9_ciciot2023_split_distribution.png
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd


class LiteV1TrainingCurvePlotter:
    """Parse Lite_V9_CICIoT2023 epoch summaries and draw training/validation curves."""

    EPOCH_RE = re.compile(
        r"EPOCH\s+(?P<epoch>\d+)\s+RESULTS:.*?"
        r"Train Loss:\s+(?P<train_loss>[0-9.]+)\s+\|\s+Train Acc:\s+(?P<train_acc>[0-9.]+).*?"
        r"Val Loss:\s+(?P<val_loss>[0-9.]+)\s+\|\s+Val Acc:\s+(?P<val_acc>[0-9.]+).*?"
        r"Val Macro-F1:\s+(?P<val_macro_f1>[0-9.]+)\s+\|\s+Val Weighted-F1:\s+(?P<val_weighted_f1>[0-9.]+).*?"
        r"Val Balanced Acc:\s+(?P<val_balanced_acc>[0-9.]+)",
        re.DOTALL,
    )

    def __init__(self, log_path: Path, output_path: Path):
        self.log_path = log_path
        self.output_path = output_path

    def parse_epochs(self) -> list[dict[str, float]]:
        text = self.log_path.read_text(errors="replace")
        rows: list[dict[str, float]] = []

        for match in self.EPOCH_RE.finditer(text):
            row: dict[str, float] = {"epoch": float(match.group("epoch"))}
            for key in (
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_weighted_f1",
                "val_balanced_acc",
            ):
                row[key] = float(match.group(key))
            rows.append(row)

        if not rows:
            raise ValueError(f"No epoch summaries found in {self.log_path}")

        return rows

    @staticmethod
    def column(rows: list[dict[str, float]], key: str) -> list[float]:
        return [row[key] for row in rows]

    def draw(self) -> Path:
        rows = self.parse_epochs()
        epochs = self.column(rows, "epoch")

        best_macro = max(rows, key=lambda row: row["val_macro_f1"])
        best_acc = max(rows, key=lambda row: row["val_acc"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Lite_V9_CICIoT2023 Training Curves", fontsize=16, fontweight="bold")

        axes[0].plot(epochs, self.column(rows, "train_loss"), label="train", linewidth=2)
        axes[0].plot(epochs, self.column(rows, "val_loss"), label="valid", linewidth=2)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, self.column(rows, "train_acc"), label="train", linewidth=2)
        axes[1].plot(epochs, self.column(rows, "val_acc"), label="valid", linewidth=2)
        axes[1].scatter(
            [best_acc["epoch"]],
            [best_acc["val_acc"]],
            color="tab:orange",
            zorder=5,
            label=f"best val acc: {best_acc['val_acc']:.4f}",
        )
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, self.column(rows, "val_macro_f1"), label="macro-F1", linewidth=2)
        axes[2].plot(epochs, self.column(rows, "val_balanced_acc"), label="balanced acc", linewidth=2)
        axes[2].plot(
            epochs,
            self.column(rows, "val_weighted_f1"),
            label="weighted-F1",
            linewidth=2,
            alpha=0.8,
        )
        axes[2].scatter(
            [best_macro["epoch"]],
            [best_macro["val_macro_f1"]],
            color="tab:red",
            zorder=5,
            label=f"best macro-F1: {best_macro['val_macro_f1']:.4f}",
        )
        axes[2].set_title("Validation Metrics")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Score")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        for ax in axes:
            ax.set_xlim(min(epochs), max(epochs))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return self.output_path


class LiteV1PredictionProbabilityPlotter:
    """Draw predicted class probability distributions grouped by true class."""

    def __init__(
        self,
        predictions_path: Path,
        output_path: Path,
        label_classes_path: Path,
        predicted_class: int = 1,
        max_points_per_true_class: int = 2000,
        random_seed: int = 42,
    ):
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.label_classes_path = label_classes_path
        self.predicted_class = predicted_class
        self.max_points_per_true_class = max_points_per_true_class
        self.random_seed = random_seed

    def load_label_names(self) -> list[str] | None:
        if not self.label_classes_path.exists():
            return None
        with self.label_classes_path.open("rb") as f:
            labels = pickle.load(f)
        return [str(label) for label in labels]

    def sample_probabilities(self) -> tuple[dict[int, list[float]], dict[int, int]]:
        random.seed(self.random_seed)
        prob_col = f"prob_class_{self.predicted_class}"
        samples: dict[int, list[float]] = {}
        counts: dict[int, int] = {}

        with self.predictions_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if prob_col not in reader.fieldnames:
                raise ValueError(f"Column {prob_col!r} was not found in {self.predictions_path}")

            for row in reader:
                true_label = int(row["true_label"])
                probability = float(row[prob_col])

                counts[true_label] = counts.get(true_label, 0) + 1
                bucket = samples.setdefault(true_label, [])

                if len(bucket) < self.max_points_per_true_class:
                    bucket.append(probability)
                else:
                    replacement_index = random.randint(0, counts[true_label] - 1)
                    if replacement_index < self.max_points_per_true_class:
                        bucket[replacement_index] = probability

        return samples, counts

    def draw(self) -> Path:
        samples, counts = self.sample_probabilities()
        label_names = self.load_label_names()

        true_classes = sorted(samples)
        y_lookup = {label: idx for idx, label in enumerate(true_classes)}

        fig_height = max(6, 0.45 * len(true_classes))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        x_values: list[float] = []
        y_values: list[float] = []
        colors: list[str] = []

        for label in true_classes:
            y_base = y_lookup[label]
            for probability in samples[label]:
                x_values.append(probability)
                y_values.append(y_base + random.uniform(-0.22, 0.22))
                colors.append("tab:orange" if label == self.predicted_class else "mediumspringgreen")

        ax.scatter(
            x_values,
            y_values,
            c=colors,
            alpha=0.28,
            s=32,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.axvline(0.5, color="gray", linewidth=1.5)
        ax.grid(axis="x", alpha=0.25)

        y_labels = []
        for label in true_classes:
            if label_names and 0 <= label < len(label_names):
                y_labels.append(f"{label}: {label_names[label]} ({counts[label]:,})")
            else:
                y_labels.append(f"Class {label} ({counts[label]:,})")

        predicted_name = (
            label_names[self.predicted_class]
            if label_names and 0 <= self.predicted_class < len(label_names)
            else f"class {self.predicted_class}"
        )

        ax.set_yticks([y_lookup[label] for label in true_classes])
        ax.set_yticklabels(y_labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("True Class")
        ax.set_title(f"Predicted class {self.predicted_class} probability per true class ({predicted_name})")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return self.output_path


class LiteV1ProbabilityHeatmapPlotter:
    """Draw mean predicted probability for every predicted class by true class."""

    def __init__(self, predictions_path: Path, output_path: Path, label_classes_path: Path):
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.label_classes_path = label_classes_path

    def load_label_names(self) -> list[str]:
        with self.label_classes_path.open("rb") as f:
            labels = pickle.load(f)
        return [str(label) for label in labels]

    def compute_mean_probabilities(self) -> tuple[list[str], list[list[float]], list[int]]:
        label_names = self.load_label_names()
        n_classes = len(label_names)
        sums = [[0.0 for _ in range(n_classes)] for _ in range(n_classes)]
        counts = [0 for _ in range(n_classes)]
        prob_cols = [f"prob_class_{idx}" for idx in range(n_classes)]

        with self.predictions_path.open(newline="") as f:
            reader = csv.DictReader(f)
            missing = [col for col in prob_cols if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing probability columns: {missing}")

            for row in reader:
                true_label = int(row["true_label"])
                counts[true_label] += 1
                for pred_class, col in enumerate(prob_cols):
                    sums[true_label][pred_class] += float(row[col])

        means = []
        for true_label in range(n_classes):
            if counts[true_label] == 0:
                means.append([0.0 for _ in range(n_classes)])
            else:
                means.append([value / counts[true_label] for value in sums[true_label]])

        return label_names, means, counts

    def draw(self) -> Path:
        label_names, means, counts = self.compute_mean_probabilities()
        n_classes = len(label_names)

        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(means, cmap="viridis", vmin=0.0, vmax=1.0)

        x_labels = [f"{idx}\n{name}" for idx, name in enumerate(label_names)]
        y_labels = [f"{idx} {name}\n({counts[idx]:,})" for idx, name in enumerate(label_names)]

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Predicted Class Probability")
        ax.set_ylabel("True Class")
        ax.set_title("Lite_V9_CICIoT2023 Mean Predicted Probability by True Class")

        for true_idx in range(n_classes):
            for pred_idx in range(n_classes):
                value = means[true_idx][pred_idx]
                if value >= 0.05 or true_idx == pred_idx:
                    text_color = "white" if value > 0.45 else "black"
                    ax.text(
                        pred_idx,
                        true_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean predicted probability")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        return self.output_path


class LiteV1SplitDistributionPlotter:
    """Draw approximate chronological train/validation/test split distribution."""

    def __init__(
        self,
        data_path: Path,
        output_path: Path,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        max_points: int = 30000,
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.valid_size = valid_size
        self.test_size = test_size
        self.max_points = max_points

    def compute_split_codes(self) -> list[int]:
        df = pd.read_pickle(self.data_path)
        train_ratio = 1.0 - self.valid_size - self.test_size
        if train_ratio <= 0:
            raise ValueError("Invalid split sizes: train ratio must be positive.")

        if "Label" not in df.columns:
            raise ValueError("Expected a Label column in the processed dataframe.")

        if "Timestamp" in df.columns:
            ordered = df.sort_values("Timestamp").reset_index(drop=True)
        else:
            ordered = df.reset_index(drop=True)

        split_codes = [-1] * len(ordered)

        for _, class_df in ordered.groupby("Label", sort=False):
            positions = class_df.index.to_list()
            n = len(positions)
            n_train = max(1, int(n * train_ratio))
            n_val = int(n * self.valid_size)
            n_test = n - n_train - n_val

            if n_test <= 0:
                n_test = 1
                if n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1

            for pos in positions[:n_train]:
                split_codes[pos] = 0
            for pos in positions[n_train:n_train + n_val]:
                split_codes[pos] = 1
            for pos in positions[n_train + n_val:]:
                split_codes[pos] = 2

        if any(code == -1 for code in split_codes):
            raise ValueError("Some rows were not assigned to a split.")

        return split_codes

    def downsample_codes(self, split_codes: list[int]) -> list[int]:
        if len(split_codes) <= self.max_points:
            return split_codes
        step = len(split_codes) / self.max_points
        return [split_codes[int(i * step)] for i in range(self.max_points)]

    def draw(self) -> Path:
        split_codes = self.compute_split_codes()
        shown_codes = self.downsample_codes(split_codes)

        fig, ax = plt.subplots(figsize=(18, 2.2))
        cmap = ListedColormap(["blue", "orange", "limegreen"])
        ax.imshow([shown_codes], aspect="auto", cmap=cmap, vmin=0, vmax=2)

        ax.set_title("Lite_V9_CICIoT2023 Approximate Split Distribution")
        ax.set_xlabel(
            f"Chronological row position, downsampled to {len(shown_codes):,} points "
            f"from {len(split_codes):,} rows"
        )
        ax.set_yticks([0])
        ax.set_yticklabels(["Rows"])

        legend_items = [
            Patch(facecolor="blue", label="Train"),
            Patch(facecolor="orange", label="Valid"),
            Patch(facecolor="limegreen", label="Test"),
        ]
        ax.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.01, 0.5))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        return self.output_path


class LiteV1LabelDistributionPlotter:
    """Draw true vs predicted label-count distribution for all classes."""

    def __init__(self, predictions_path: Path, output_path: Path, label_classes_path: Path):
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.label_classes_path = label_classes_path

    def load_label_names(self) -> list[str]:
        with self.label_classes_path.open("rb") as f:
            labels = pickle.load(f)
        return [str(label) for label in labels]

    def count_labels(self) -> tuple[list[str], list[int], list[int]]:
        label_names = self.load_label_names()
        n_classes = len(label_names)
        true_counts = [0 for _ in range(n_classes)]
        pred_counts = [0 for _ in range(n_classes)]

        with self.predictions_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_label = int(row["true_label"])
                pred_label = int(row["predicted_label"])
                true_counts[true_label] += 1
                pred_counts[pred_label] += 1

        return label_names, true_counts, pred_counts

    def draw(self) -> Path:
        label_names, true_counts, pred_counts = self.count_labels()
        n_classes = len(label_names)
        max_count = max(max(true_counts), max(pred_counts))

        fig_height = max(8, 0.78 * n_classes)
        fig, ax = plt.subplots(figsize=(18, fig_height))

        true_color = "#6258f2"
        pred_color = "#9aaabd"
        row_bg = "#f4f7fb"
        row_edge = "#e9eef5"

        y_base = [idx * 2.0 for idx in range(n_classes)]
        true_y = [y - 0.26 for y in y_base]
        pred_y = [y + 0.26 for y in y_base]

        for y in y_base:
            ax.barh(
                y,
                max_count * 1.06,
                height=1.42,
                color=row_bg,
                edgecolor=row_edge,
                linewidth=0.8,
                zorder=0,
            )

        ax.barh(true_y, true_counts, height=0.34, color=true_color, label="True Labels", zorder=3)
        ax.barh(pred_y, pred_counts, height=0.34, color=pred_color, label="Predicted Labels", zorder=3)

        label_offset = max_count * 0.008
        for y, count in zip(true_y, true_counts):
            ax.text(
                count + label_offset,
                y,
                f"{count:,}",
                va="center",
                ha="left",
                color="#4a5568",
                fontsize=10,
                fontweight="bold",
                family="monospace",
            )

        for y, count in zip(pred_y, pred_counts):
            ax.text(
                count + label_offset,
                y,
                f"{count:,}",
                va="center",
                ha="left",
                color="#4a5568",
                fontsize=10,
                fontweight="bold",
                family="monospace",
            )

        y_labels = [f"{idx}: {name}" for idx, name in enumerate(label_names)]
        ax.set_yticks(y_base)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max_count * 1.16)
        ax.set_xlabel("Sample Count", fontsize=11)
        ax.set_title("Lite_V9_CICIoT2023 Label Distribution Analysis", fontsize=24, fontweight="bold", loc="left", pad=28)
        ax.grid(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", length=0, labelsize=10, pad=10)

        legend_items = [
            Patch(facecolor=true_color, edgecolor=true_color, label="True Labels"),
            Patch(facecolor=pred_color, edgecolor=pred_color, label="Predicted Labels"),
        ]
        ax.legend(
            handles=legend_items,
            loc="upper left",
            bbox_to_anchor=(0, 1.035),
            ncols=2,
            frameon=False,
            fontsize=11,
            handlelength=1.0,
            handleheight=1.0,
        )

        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        return self.output_path


def resolve_path(project_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_dir / path


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Draw Lite_V9_CICIoT2023 training and evaluation plots.")
    parser.add_argument(
        "--plot",
        choices=("curves", "probability", "heatmap", "split", "labels", "both"),
        default="curves",
        help="Which plot to create.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=project_dir / "outputs" / "logs" / "lite_v9_ciciot2023_step_train_16913_20260616_010550.txt",
        help="Training log path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "outputs" / "results" / "lite_v9_ciciot2023_training_curves.png",
        help="Output PNG path for training curves.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_dir / "outputs" / "results" / "predictions.csv",
        help="Predictions CSV path.",
    )
    parser.add_argument(
        "--label-classes",
        type=Path,
        default=project_dir / "outputs" / "data" / "label_classes.pkl",
        help="Label class metadata path.",
    )
    parser.add_argument(
        "--predicted-class",
        type=int,
        default=1,
        help="Class index whose predicted probability should be plotted.",
    )
    parser.add_argument(
        "--max-points-per-true-class",
        type=int,
        default=2000,
        help="Maximum sampled points per true class.",
    )
    parser.add_argument(
        "--prob-output",
        type=Path,
        default=project_dir / "outputs" / "results" / "lite_v9_ciciot2023_predicted_class_1_probability_by_true_class.png",
        help="Output PNG path for probability plot.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=project_dir / "outputs" / "results" / "lite_v9_ciciot2023_probability_heatmap_by_true_class.png",
        help="Output PNG path for probability heatmap.",
    )
    parser.add_argument(
        "--split-data",
        type=Path,
        default=project_dir / "outputs" / "data" / "step_2_Processed_Data.pkl",
        help="Processed dataframe used to recreate approximate split distribution.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=project_dir / "outputs" / "results" / "lite_v9_ciciot2023_split_distribution.png",
        help="Output PNG path for split distribution plot.",
    )
    parser.add_argument(
        "--split-max-points",
        type=int,
        default=30000,
        help="Number of points to display in the split distribution barcode.",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=project_dir / "outputs" / "results" / "lite_v9_ciciot2023_label_distribution_analysis.png",
        help="Output PNG path for true/predicted label distribution plot.",
    )
    args = parser.parse_args()

    log_path = resolve_path(project_dir, args.log)
    output_path = resolve_path(project_dir, args.output)
    predictions_path = resolve_path(project_dir, args.predictions)
    label_classes_path = resolve_path(project_dir, args.label_classes)
    prob_output_path = resolve_path(project_dir, args.prob_output)
    heatmap_output_path = resolve_path(project_dir, args.heatmap_output)
    split_data_path = resolve_path(project_dir, args.split_data)
    split_output_path = resolve_path(project_dir, args.split_output)
    label_output_path = resolve_path(project_dir, args.label_output)

    if args.plot in {"curves", "both"}:
        plotter = LiteV1TrainingCurvePlotter(log_path=log_path, output_path=output_path)
        saved_path = plotter.draw()
        print(f"Saved training curves to: {saved_path}")

    if args.plot in {"probability", "both"}:
        prob_plotter = LiteV1PredictionProbabilityPlotter(
            predictions_path=predictions_path,
            output_path=prob_output_path,
            label_classes_path=label_classes_path,
            predicted_class=args.predicted_class,
            max_points_per_true_class=args.max_points_per_true_class,
        )
        saved_path = prob_plotter.draw()
        print(f"Saved probability plot to: {saved_path}")

    if args.plot in {"heatmap", "both"}:
        heatmap_plotter = LiteV1ProbabilityHeatmapPlotter(
            predictions_path=predictions_path,
            output_path=heatmap_output_path,
            label_classes_path=label_classes_path,
        )
        saved_path = heatmap_plotter.draw()
        print(f"Saved probability heatmap to: {saved_path}")

    if args.plot in {"split", "both"}:
        split_plotter = LiteV1SplitDistributionPlotter(
            data_path=split_data_path,
            output_path=split_output_path,
            max_points=args.split_max_points,
        )
        saved_path = split_plotter.draw()
        print(f"Saved split distribution plot to: {saved_path}")

    if args.plot in {"labels", "both"}:
        label_plotter = LiteV1LabelDistributionPlotter(
            predictions_path=predictions_path,
            output_path=label_output_path,
            label_classes_path=label_classes_path,
        )
        saved_path = label_plotter.draw()
        print(f"Saved label distribution plot to: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
