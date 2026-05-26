from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    CLASS_LABELS,
    CONFUSION_MATRIX_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_SELECTION_PATH,
    SAMPLE_ARTICLES_PATH,
    TOP_TERMS_PATH,
    VALIDATION_METRICS_PATH,
)


def load_bundle(path: Path = MODEL_BUNDLE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python scripts/train_model.py` first."
        )
    return joblib.load(path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_input_frame(title: str, body: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [title.strip()],
            "body": [body.strip()],
        }
    )


def confidence_from_margin(margin: float) -> float:
    return float(1.0 / (1.0 + math.exp(-abs(margin))))


def batch_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Uploaded file is empty.")

    columns = {column.lower().strip(): column for column in frame.columns}
    if "title" not in columns:
        raise ValueError("CSV must include a `title` column.")

    body_key = "body" if "body" in columns else "text" if "text" in columns else None
    if body_key is None:
        raise ValueError("CSV must include a `body` column or a `text` column.")

    title_column = columns["title"]
    body_column = columns[body_key]
    prepared = pd.DataFrame(
        {
            "title": frame[title_column].fillna("").astype(str).str.strip(),
            "body": frame[body_column].fillna("").astype(str).str.strip(),
        }
    )
    return prepared


def score_frame(pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(frame), dtype=float)

    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return (probabilities[:, 1] * 2.0) - 1.0

    predictions = np.asarray(pipeline.predict(frame), dtype=float)
    return predictions


def _pack_contributions(feature_names, values, indices, limit: int = 8) -> list[dict[str, str | float]]:
    rows = []
    seen = set()
    for idx in indices:
        contribution = float(values[idx])
        if contribution == 0:
            continue
        feature = feature_names[idx].replace("title__", "title: ").replace("body__", "body: ")
        feature_key = feature.split(": ", 1)[-1]
        if feature_key in seen:
            continue
        seen.add(feature_key)
        rows.append({"feature": feature, "contribution": contribution})
        if len(rows) >= limit:
            break
    return rows


def explain_prediction(bundle: dict, title: str, body: str, limit: int = 8) -> dict[str, list[dict[str, str | float]]]:
    pipeline = bundle["pipeline"]
    frame = build_input_frame(title, body)
    vector = pipeline.named_steps["features"].transform(frame)
    classifier = pipeline.named_steps["clf"]
    feature_names = pipeline.named_steps["features"].get_feature_names_out()

    if hasattr(classifier, "coef_"):
        weights = np.asarray(classifier.coef_).ravel()
    elif hasattr(classifier, "feature_log_prob_"):
        weights = np.asarray(classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]).ravel()
    else:
        return {"supporting_fake": [], "supporting_real": []}

    contributions = vector.multiply(weights).toarray().ravel()

    supporting_fake = _pack_contributions(feature_names, contributions, contributions.argsort()[::-1], limit)
    supporting_real = _pack_contributions(feature_names, contributions, contributions.argsort(), limit)
    return {
        "supporting_fake": supporting_fake,
        "supporting_real": supporting_real,
    }


def predict_article(bundle: dict, title: str, body: str) -> dict:
    pipeline = bundle["pipeline"]
    frame = build_input_frame(title, body)
    label = int(pipeline.predict(frame)[0])
    margin = float(score_frame(pipeline, frame)[0])
    confidence = confidence_from_margin(margin)
    explanation = explain_prediction(bundle, title, body)

    return {
        "label": label,
        "label_name": CLASS_LABELS[label],
        "decision_score": margin,
        "confidence": confidence,
        "explanation": explanation,
    }


def predict_batch(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    inputs = batch_input_frame(frame)
    labels = pipeline.predict(inputs)
    margins = score_frame(pipeline, inputs)

    results = frame.copy()
    results["predicted_label"] = [CLASS_LABELS[int(label)] for label in labels]
    results["predicted_class"] = [int(label) for label in labels]
    results["decision_score"] = np.asarray(margins, dtype=float)
    results["confidence"] = [confidence_from_margin(float(score)) for score in margins]
    return results


def load_dashboard_payload() -> dict:
    return {
        "metrics": load_json(VALIDATION_METRICS_PATH, {}),
        "selection": load_json(MODEL_SELECTION_PATH, []),
        "top_terms": load_json(TOP_TERMS_PATH, {}),
        "samples": load_json(SAMPLE_ARTICLES_PATH, []),
        "confusion": load_json(CONFUSION_MATRIX_PATH, {}),
    }
