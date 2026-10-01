"""Train and evaluate the Random Forest using video-level validation.

Run:
    python -m src.train_model

Evaluation uses GroupKFold so complete videos are held out.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from . import config
from .analyze import merge_events
from .features import FEATURE_COLUMNS
from .labels import intervals_for, parse_anomaly_file


def make_rf(seed=42):
    """
    Random Forest model.

    We use a slightly more conservative class weighting
    than balanced_subsample because the previous model
    was predicting too many anomaly windows.
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight={0: 1.3, 1: 1.0},
        random_state=seed,
        n_jobs=-1,
    )


def test_thresholds(y, probabilities):
    """
    Test multiple classification thresholds.

    Returns the threshold that gives the highest
    validation accuracy.
    """

    thresholds = np.arange(0.20, 0.81, 0.05)

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(
            y, predictions, zero_division=0
        )
        recall = recall_score(
            y, predictions, zero_division=0
        )
        f1 = f1_score(
            y, predictions, zero_division=0
        )

        results.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    # Select the threshold with the highest accuracy.
    # If two thresholds have similar accuracy,
    # prefer the one with better F1.
    best_row = results_df.sort_values(
        ["accuracy", "f1"],
        ascending=[False, False],
    ).iloc[0]

    return float(best_row["threshold"]), results_df


def event_metrics(df, flag_col, labels, min_windows):

    tp = 0
    fp = 0
    fn = 0
    hours = 0.0

    for vid, group in df.groupby("video_id"):

        hours += (
            group["t_end"].max()
            - group["t_start"].min()
        ) / 3600

        ground_truth = intervals_for(
            labels,
            int(vid)
        )

        events = merge_events(
            group.assign(
                flag=group[flag_col],
                score=group["score"],
                reason="-",
            ),
            min_windows,
        )

        hit = [False] * len(ground_truth)

        for event in events:

            matched = False

            for k, (start, end) in enumerate(
                ground_truth
            ):

                if (
                    event["start"] < end
                    and event["end"] > start
                ):
                    hit[k] = True
                    matched = True

            if matched:
                tp += 1
            else:
                fp += 1

        fn += hit.count(False)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_alarms_per_hour": (
            fp / hours if hours else 0.0
        ),
        "events_tp": tp,
        "events_fp": fp,
        "events_fn": fn,
    }


def main():

    if not config.FEATURES_CSV.exists():

        raise SystemExit(
            "dataset/features.csv not found.\n"
            "Run feature extraction first."
        )

    df = pd.read_csv(
        config.FEATURES_CSV
    )

    labels = parse_anomaly_file()

    df = (
        df.dropna(subset=["label"])
        .reset_index(drop=True)
    )

    y = df["label"].astype(int).to_numpy()

    groups = df["video_id"].to_numpy()

    X = df[FEATURE_COLUMNS]

    n_videos = len(
        np.unique(groups)
    )

    print(
        f"{len(df)} windows | "
        f"{n_videos} videos | "
        f"normal={(y == 0).sum()} "
        f"anomaly={(y == 1).sum()}"
    )

    if len(np.unique(y)) < 2:

        raise SystemExit(
            "Need both normal and anomaly windows."
        )

    if n_videos < 3:

        print(
            "WARNING: fewer than 3 videos. "
            "Validation is not meaningful."
        )

        n_splits = 0

    else:

        n_splits = min(
            5,
            n_videos
        )

    metrics = {
        "n_windows": int(len(df)),
        "n_videos": int(n_videos),
    }

    threshold = 0.5

    if n_splits:

        # Out-of-fold probabilities.
        oof = np.zeros(
            len(df)
        )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(
                X,
                y,
                groups
            ),
            start=1,
        ):

            print(
                f"Training fold {fold}/{n_splits}..."
            )

            model = make_rf()

            model.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            oof[test_idx] = (
                model.predict_proba(
                    X.iloc[test_idx]
                )[:, 1]
            )

        # Test thresholds.
        threshold, threshold_results = test_thresholds(
            y,
            oof
        )

        print("\n=== Threshold comparison ===")

        print(
            threshold_results.to_string(
                index=False,
                formatters={
                    "accuracy": "{:.3f}".format,
                    "precision": "{:.3f}".format,
                    "recall": "{:.3f}".format,
                    "f1": "{:.3f}".format,
                },
            )
        )

        print(
            f"\nSelected threshold: {threshold:.2f}"
        )

        predictions = (
            oof >= threshold
        ).astype(int)

        metrics.update(
            {
                "cv": (
                    f"GroupKFold by video, "
                    f"{n_splits} folds"
                ),
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y,
                    predictions
                ),
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
            }
        )

        # Event-level evaluation.
        event_df = df.assign(
            score=oof,
            flag=predictions.astype(bool),
        )

        metrics.update(
            event_metrics(
                event_df,
                "flag",
                labels,
                config.MIN_EVENT_WINDOWS,
            )
        )

        # Save confusion matrix.
        config.OUTPUT_DIR.mkdir(
            exist_ok=True
        )

        figure, axis = plt.subplots(
            figsize=(5, 5)
        )

        ConfusionMatrixDisplay(
            confusion_matrix(
                y,
                predictions,
                labels=[0, 1],
            ),
            display_labels=[
                "normal",
                "anomaly",
            ],
        ).plot(ax=axis)

        axis.set_title(
            "Out-of-fold confusion matrix"
        )

        figure.tight_layout()

        figure.savefig(
            config.OUTPUT_DIR
            / "confusion_matrix.png",
            dpi=150,
        )

        plt.close(figure)

        # Save threshold comparison.
        threshold_results.to_csv(
            config.OUTPUT_DIR
            / "threshold_results.csv",
            index=False,
        )

    # Train final model on ALL available videos.
    print(
        "\nTraining final model on all videos..."
    )

    final_model = make_rf()

    final_model.fit(
        X,
        y
    )

    # Feature importance.
    importance = pd.Series(
        final_model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    metrics["top_features"] = (
        importance
        .head(10)
        .round(3)
        .to_dict()
    )

    settings = {
        "win_sec": config.WIN_SEC,
        "hop_sec": config.HOP_SEC,
        "stride": config.STRIDE,
        "imgsz": config.IMGSZ,
        "conf": config.CONF,
        "yolo": config.DEFAULT_YOLO,
    }

    metadata_path = (
        config.FEATURES_CSV.with_suffix(
            ".json"
        )
    )

    if metadata_path.exists():

        settings.update(
            json.loads(
                metadata_path.read_text()
            )
        )

    # Save model.
    config.MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
            "metrics": metrics,
            **settings,
        },
        config.MODEL_PATH,
    )

    config.OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    (
        config.OUTPUT_DIR
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2
        )
    )

    # Final output.
    print(
        "\n======================================"
    )

    print(
        "FINAL VALIDATION RESULTS"
    )

    print(
        "======================================"
    )

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.3f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        "\nModel saved ->"
    )

    print(
        config.MODEL_PATH
    )

    print(
        "\nTop features:"
    )

    print(
        importance.head(10)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()