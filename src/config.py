from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "News.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "fake_news_detector.joblib"
VALIDATION_METRICS_PATH = ARTIFACTS_DIR / "validation_metrics.json"
MODEL_SELECTION_PATH = ARTIFACTS_DIR / "model_selection.json"
TOP_TERMS_PATH = ARTIFACTS_DIR / "top_terms.json"
SAMPLE_ARTICLES_PATH = ARTIFACTS_DIR / "sample_articles.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_BODY_LENGTH = 80
CLASS_LABELS = {
    0: "Likely real",
    1: "Likely fake",
}
