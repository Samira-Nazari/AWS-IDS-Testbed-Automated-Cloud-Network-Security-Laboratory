#!/usr/bin/env python3
"""Focused distribution diagnostics for confused Lite_V9_CICIoT2023 classes.

This script reads the saved Lite_V9_CICIoT2023 windows/predictions and creates plots for
classes 1, 2, 3, and 11 without changing any training artifacts.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "outputs" / "data"
RESULTS_DIR = PROJECT_DIR / "outputs" / "results"
OUT_DIR = RESULTS_DIR / "class_confusion_diagnostics"

FOCUS_CLASSES = [1, 2, 3, 11]
PAIR_CLASSES = [(1, 2), (3, 11)]
TRAIN_SAMPLE_PER_CLASS = 4000
PCA_SAMPLE_PER_CLASS = 2500
MISTAKE_SAMPLE_PER_GROUP = 2500
RANDOM_SEED = 42


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def class_label(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return f"{class_id}: {class_names[class_id]}"
    return str(class_id)


def sample_indices(y: np.ndarray, class_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(y == class_id)
    if len(idx) == 0:
        return idx
    size = min(n, len(idx))
    return np.sort(rng.choice(idx, size=size, replace=False))


def summarize_windows(
    x_memmap: np.ndarray,
    indices: np.ndarray,
    feature_names: list[str],
    class_id: int | None = None,
    group_name: str | None = None,
) -> pd.DataFrame:
    """Convert sampled windows to per-window feature means."""
    if len(indices) == 0:
        return pd.DataFrame()
    x = np.asarray(x_memmap[indices], dtype=np.float32)
    x_mean = x.mean(axis=1)
    df = pd.DataFrame(x_mean, columns=feature_names)
    df["sample_index"] = indices
    if class_id is not None:
        df["class_id"] = class_id
    if group_name is not None:
        df["group"] = group_name
    return df


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Small dependency-free Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def pair_feature_stats(sample_df: pd.DataFrame, feature_names: list[str], a: int, b: int) -> pd.DataFrame:
    rows = []
    left = sample_df[sample_df["class_id"] == a]
    right = sample_df[sample_df["class_id"] == b]
    for feature in feature_names:
        x = left[feature].to_numpy()
        y = right[feature].to_numpy()
        pooled = np.sqrt((np.var(x) + np.var(y)) / 2.0)
        effect = 0.0 if pooled == 0 else abs(np.mean(x) - np.mean(y)) / pooled
        rows.append(
            {
                "pair": f"{a}_vs_{b}",
                "feature": feature,
                f"class_{a}_mean": np.mean(x),
                f"class_{b}_mean": np.mean(y),
                f"class_{a}_median": np.median(x),
                f"class_{b}_median": np.median(y),
                "abs_standardized_mean_diff": effect,
                "ks_statistic": ks_statistic(x, y),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def save_confusion_outputs(pred_df: pd.DataFrame, class_names: list[str]) -> None:
    focus_pred = pred_df[pred_df["true_label"].isin(FOCUS_CLASSES)].copy()
    confusion = pd.crosstab(focus_pred["true_label"], focus_pred["predicted_label"])
    nonzero_cols = [c for c in confusion.columns if confusion[c].sum() > 0]
    confusion = confusion[nonzero_cols]
    confusion.to_csv(OUT_DIR / "focus_confusion_counts.csv")

    plt.figure(figsize=(12, 4.8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="mako")
    plt.title("Test Confusion Counts for Focus True Classes")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.yticks(
        ticks=np.arange(len(confusion.index)) + 0.5,
        labels=[class_label(int(i), class_names) for i in confusion.index],
        rotation=0,
    )
    plt.xticks(
        ticks=np.arange(len(confusion.columns)) + 0.5,
        labels=[class_label(int(i), class_names) for i in confusion.columns],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "focus_confusion_heatmap.png", dpi=180)
    plt.close()

    mistake_rows = []
    for true_label in FOCUS_CLASSES:
        sub = focus_pred[focus_pred["true_label"] == true_label]
        counts = Counter(sub["predicted_label"])
        for pred_label, count in counts.most_common():
            mistake_rows.append(
                {
                    "true_label": true_label,
                    "true_name": class_names[true_label],
                    "predicted_label": pred_label,
                    "predicted_name": class_names[pred_label],
                    "count": count,
                    "rate_within_true_class": count / len(sub),
                }
            )
    pd.DataFrame(mistake_rows).to_csv(OUT_DIR / "focus_prediction_counts.csv", index=False)


def save_class_count_outputs(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> None:
    rows = []
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = Counter(y.tolist())
        for class_id in FOCUS_CLASSES:
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "count": counts.get(class_id, 0),
                }
            )
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(OUT_DIR / "focus_class_counts_by_split.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=counts_df, x="class_name", y="count", hue="split")
    plt.title("Focus Class Counts by Split")
    plt.xlabel("Class")
    plt.ylabel("Window count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "focus_class_counts_by_split.png", dpi=180)
    plt.close()


def save_feature_distribution_plots(sample_df: pd.DataFrame, feature_names: list[str], class_names: list[str]) -> None:
    # Global high-variance features among focus classes.
    means = sample_df.groupby("class_id")[feature_names].mean()
    top_features = means.var(axis=0).sort_values(ascending=False).head(8).index.tolist()
    melted = sample_df.melt(
        id_vars=["class_id"],
        value_vars=top_features,
        var_name="feature",
        value_name="window_mean",
    )
    melted["class"] = melted["class_id"].map(lambda c: class_label(int(c), class_names))

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=melted, x="feature", y="window_mean", hue="class", showfliers=False)
    plt.title("Top Focus-Class Feature Distributions - Window Mean")
    plt.xlabel("Feature")
    plt.ylabel("Mean value over 30-step window")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_focus_feature_boxplots.png", dpi=180)
    plt.close()

    all_pair_stats = []
    for a, b in PAIR_CLASSES:
        stats = pair_feature_stats(sample_df, feature_names, a, b)
        all_pair_stats.append(stats)
        top_pair_features = stats.head(6)["feature"].tolist()
        pair_df = sample_df[sample_df["class_id"].isin([a, b])].copy()
        pair_melted = pair_df.melt(
            id_vars=["class_id"],
            value_vars=top_pair_features,
            var_name="feature",
            value_name="window_mean",
        )
        pair_melted["class"] = pair_melted["class_id"].map(lambda c: class_label(int(c), class_names))

        plt.figure(figsize=(12, 6.5))
        sns.violinplot(data=pair_melted, x="feature", y="window_mean", hue="class", cut=0, inner="quartile")
        plt.title(f"Pair Distribution: {class_label(a, class_names)} vs {class_label(b, class_names)}")
        plt.xlabel("Most different sampled features")
        plt.ylabel("Mean value over 30-step window")
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pair_{a}_vs_{b}_top_feature_violins.png", dpi=180)
        plt.close()

    pd.concat(all_pair_stats, ignore_index=True).to_csv(OUT_DIR / "pair_feature_separability.csv", index=False)


def save_pca_plot(sample_df: pd.DataFrame, feature_names: list[str], class_names: list[str]) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    pieces = []
    for class_id in FOCUS_CLASSES:
        sub = sample_df[sample_df["class_id"] == class_id]
        take = min(PCA_SAMPLE_PER_CLASS, len(sub))
        pieces.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    pca_df = pd.concat(pieces, ignore_index=True)

    x = StandardScaler().fit_transform(pca_df[feature_names].to_numpy())
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["class"] = pca_df["class_id"].map(lambda c: class_label(int(c), class_names))

    plt.figure(figsize=(8, 6.5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class", s=12, alpha=0.4, linewidth=0)
    plt.title("PCA of Sampled Training Windows - Focus Classes")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_focus_classes_train.png", dpi=180)
    plt.close()


def save_mistake_conditioned_plots(
    x_test: np.ndarray,
    pred_df: pd.DataFrame,
    feature_names: list[str],
    class_names: list[str],
    pair_stats_df: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    groups_by_pair = {
        (1, 2): [
            ("true1_pred1", 1, 1),
            ("true1_pred2", 1, 2),
            ("true2_pred2", 2, 2),
            ("true2_pred1", 2, 1),
        ],
        (3, 11): [
            ("true3_pred3", 3, 3),
            ("true3_pred11", 3, 11),
            ("true11_pred11", 11, 11),
            ("true11_pred3", 11, 3),
            ("true3_pred4", 3, 4),
            ("true11_pred4", 11, 4),
        ],
    }

    for pair, groups in groups_by_pair.items():
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        top_features = (
            pair_stats_df[pair_stats_df["pair"] == pair_name]
            .head(5)["feature"]
            .tolist()
        )
        pieces = []
        for group_name, true_id, pred_id in groups:
            idx = pred_df.index[
                (pred_df["true_label"] == true_id) & (pred_df["predicted_label"] == pred_id)
            ].to_numpy()
            if len(idx) == 0:
                continue
            take = min(MISTAKE_SAMPLE_PER_GROUP, len(idx))
            sampled = np.sort(rng.choice(idx, size=take, replace=False))
            pieces.append(summarize_windows(x_test, sampled, feature_names, group_name=group_name))
        if not pieces:
            continue
        mistake_df = pd.concat(pieces, ignore_index=True)
        mistake_df.to_csv(OUT_DIR / f"pair_{pair_name}_mistake_sample_window_means.csv", index=False)

        melted = mistake_df.melt(
            id_vars=["group"],
            value_vars=top_features,
            var_name="feature",
            value_name="window_mean",
        )
        plt.figure(figsize=(13, 7))
        sns.boxplot(data=melted, x="feature", y="window_mean", hue="group", showfliers=False)
        plt.title(
            "Correct vs Confused Test Windows: "
            f"{class_label(pair[0], class_names)} / {class_label(pair[1], class_names)}"
        )
        plt.xlabel("Feature")
        plt.ylabel("Mean value over 30-step window")
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pair_{pair_name}_mistake_boxplots.png", dpi=180)
        plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    rng = np.random.default_rng(RANDOM_SEED)

    feature_names = list(load_pickle(DATA_DIR / "selected_features.pkl"))
    class_names = list(load_pickle(DATA_DIR / "label_classes.pkl"))

    y_train = np.load(DATA_DIR / "y_train.npy", mmap_mode="r")
    y_val = np.load(DATA_DIR / "y_val.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "y_test.npy", mmap_mode="r")
    x_train = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    pred_df = pd.read_csv(RESULTS_DIR / "predictions.csv")

    save_class_count_outputs(y_train, y_val, y_test, class_names)
    save_confusion_outputs(pred_df, class_names)

    train_samples = []
    for class_id in FOCUS_CLASSES:
        idx = sample_indices(y_train, class_id, TRAIN_SAMPLE_PER_CLASS, rng)
        train_samples.append(summarize_windows(x_train, idx, feature_names, class_id=class_id))
    sample_df = pd.concat(train_samples, ignore_index=True)
    sample_df.to_csv(OUT_DIR / "train_focus_sample_window_means.csv", index=False)

    save_feature_distribution_plots(sample_df, feature_names, class_names)
    save_pca_plot(sample_df, feature_names, class_names)

    pair_stats_df = pd.read_csv(OUT_DIR / "pair_feature_separability.csv")
    save_mistake_conditioned_plots(x_test, pred_df, feature_names, class_names, pair_stats_df, rng)

    print(f"Diagnostics saved to: {OUT_DIR}")
    print("Key files:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
