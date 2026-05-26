#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_bundle, load_dashboard_payload, predict_article, predict_batch  # noqa: E402


def main() -> int:
    bundle = load_bundle()
    payload = load_dashboard_payload()
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError("No sample articles found. Run `python scripts/train_model.py` first.")

    article = samples[0]
    result = predict_article(bundle, article["title"], article["body"])
    print("Title:", article["title"][:100])
    print("Prediction:", result["label_name"])
    print("Confidence:", round(result["confidence"], 4))
    print("Decision score:", round(result["decision_score"], 4))
    print("Top fake signal count:", len(result["explanation"]["supporting_fake"]))

    batch_results = predict_batch(
        bundle,
        pd.DataFrame(
            [
                {"title": article["title"], "body": article["body"]},
                {"title": "Local council confirms park renovation budget", "body": "City officials approved the next phase of the renovation after the public review meeting and budget vote."},
            ]
        ),
    )
    print("Batch rows scored:", len(batch_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
