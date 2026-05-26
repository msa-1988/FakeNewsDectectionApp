#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_SELECTION_PATH,
    SAMPLE_ARTICLES_PATH,
    TOP_TERMS_PATH,
    VALIDATION_METRICS_PATH,
)
from src.dataset import load_dataset
from src.modeling import (
    build_sample_articles,
    evaluate_candidates,
    evaluate_holdout,
    extract_global_top_terms,
    metadata_payload,
    to_pretty_json,
)


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_dataset()
    leaderboard, best_name, best_pipeline, split_data = evaluate_candidates(frame)
    metrics, confusion, prediction_frame = evaluate_holdout(best_pipeline, split_data)
    top_terms = extract_global_top_terms(best_pipeline)
    samples = build_sample_articles(prediction_frame)
    metadata = metadata_payload(frame, best_name)

    bundle = {
        "pipeline": best_pipeline,
        "metadata": metadata,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    MODEL_SELECTION_PATH.write_text(
        json.dumps(leaderboard.round(6).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    VALIDATION_METRICS_PATH.write_text(to_pretty_json(metrics), encoding="utf-8")
    TOP_TERMS_PATH.write_text(to_pretty_json(top_terms), encoding="utf-8")
    SAMPLE_ARTICLES_PATH.write_text(to_pretty_json(samples), encoding="utf-8")
    CONFUSION_MATRIX_PATH.write_text(to_pretty_json(confusion), encoding="utf-8")

    print("Best model:", best_name)
    print("Saved:", MODEL_BUNDLE_PATH)
    print("Metrics:", VALIDATION_METRICS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
