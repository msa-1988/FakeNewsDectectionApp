from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import CLASS_LABELS, RANDOM_STATE, TEST_SIZE
from .dataset import model_frame


def build_feature_stack() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "title",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=15000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
                "title",
            ),
            (
                "body",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=70000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
                "body",
            ),
        ],
        sparse_threshold=0.3,
    )


def build_candidates() -> dict[str, Pipeline]:
    feature_stack = build_feature_stack()
    return {
        "linear_svc": Pipeline(
            [
                ("features", feature_stack),
                ("clf", LinearSVC(C=1.5)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("features", build_feature_stack()),
                ("clf", LogisticRegression(max_iter=1200, C=4.0, solver="liblinear")),
            ]
        ),
        "complement_nb": Pipeline(
            [
                ("features", build_feature_stack()),
                ("clf", ComplementNB(alpha=0.3)),
            ]
        ),
    }


def evaluate_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, Pipeline, tuple]:
    X = model_frame(frame)
    y = frame["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipeline in build_candidates().items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=1,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
            },
        )
        result = {
            "model": name,
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
            "cv_precision_macro_mean": float(scores["test_precision_macro"].mean()),
            "cv_recall_macro_mean": float(scores["test_recall_macro"].mean()),
        }
        results.append(result)

    leaderboard = pd.DataFrame(results).sort_values(
        by=["cv_f1_macro_mean", "cv_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    best_name = str(leaderboard.loc[0, "model"])
    best_pipeline = build_candidates()[best_name]
    best_pipeline.fit(X_train, y_train)

    return leaderboard, best_name, best_pipeline, (X_train, X_test, y_train, y_test)


def evaluate_holdout(
    pipeline: Pipeline,
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> tuple[dict, dict, pd.DataFrame]:
    _, X_test, _, y_test = split_data
    predictions = pipeline.predict(X_test)
    decision_scores = pipeline.decision_function(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "test_roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "test_size": int(len(y_test)),
    }

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    confusion = {
        "labels": [CLASS_LABELS[0], CLASS_LABELS[1]],
        "matrix": cm.tolist(),
    }

    prediction_frame = X_test.copy()
    prediction_frame["label"] = y_test.to_numpy()
    prediction_frame["prediction"] = predictions
    prediction_frame["decision_score"] = decision_scores

    return metrics, confusion, prediction_frame


def extract_global_top_terms(pipeline: Pipeline, top_n: int = 20) -> dict[str, list[dict[str, float | str]]]:
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["clf"]
    feature_names = features.get_feature_names_out()
    coefficients = classifier.coef_.ravel()

    top_fake = np.argsort(coefficients)[-top_n:][::-1]
    top_real = np.argsort(coefficients)[:top_n]

    def pack(indices) -> list[dict[str, float | str]]:
        rows = []
        for idx in indices:
            rows.append(
                {
                    "feature": feature_names[idx].replace("title__", "title: ").replace("body__", "body: "),
                    "weight": float(coefficients[idx]),
                }
            )
        return rows

    return {
        "fake_signals": pack(top_fake),
        "real_signals": pack(top_real),
    }


def build_sample_articles(prediction_frame: pd.DataFrame, top_n: int = 2) -> list[dict[str, str | int | float]]:
    samples = []
    for label in [1, 0]:
        subset = prediction_frame[prediction_frame["label"] == label].copy()
        subset = subset[
            (subset["body"].str.len() >= 500)
            & (subset["title"].str.len() >= 25)
        ]
        subset["abs_margin"] = subset["decision_score"].abs()
        subset = subset.sort_values(by="abs_margin", ascending=False).head(top_n)
        for _, row in subset.iterrows():
            samples.append(
                {
                    "label": int(row["label"]),
                    "prediction": int(row["prediction"]),
                    "decision_score": float(row["decision_score"]),
                    "title": row["title"],
                    "body": row["body"][:1600],
                }
            )
    return samples


def metadata_payload(frame: pd.DataFrame, best_name: str) -> dict:
    label_counts = frame["label"].value_counts().to_dict()
    return {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(frame)),
        "class_balance": {
            CLASS_LABELS[int(label)]: int(count) for label, count in label_counts.items()
        },
        "features": [
            "Title word n-grams",
            "Body word n-grams",
            "Linear margin scoring",
        ],
        "notes": [
            "Author is excluded from training to reduce source-name leakage.",
            "Validation uses a stratified holdout split plus 3-fold cross-validation on the training partition.",
        ],
    }


def to_pretty_json(payload: dict | list) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)
