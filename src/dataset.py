from __future__ import annotations

import html
import re

import pandas as pd

from .config import DATASET_PATH, MIN_BODY_LENGTH

MONTH_PATTERN = (
    r"january|february|march|april|may|june|july|august|september|october|november|december"
)
SOURCE_SUFFIX_PATTERN = re.compile(
    r"\s*-\s*(the new york times|breitbart|reuters|ap|cnn|fox news|the guardian|the washington post)\s*$",
    flags=re.IGNORECASE,
)
NOISE_PATTERN = re.compile(
    rf"\b(?:{MONTH_PATTERN})\b|\b(?:19|20)\d{{2}}\b|\b(?:subscribe|share|twitter|facebook|source|image courtesy)\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    value = "" if value is None else str(value)
    value = html.unescape(value)
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"\([^)]*image[^)]*\)", " ", value, flags=re.IGNORECASE)
    value = NOISE_PATTERN.sub(" ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_title(value: str) -> str:
    value = normalize_text(value)
    value = SOURCE_SUFFIX_PATTERN.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    return prepare_dataset(dataset)


def prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    expected = {"title", "text", "author", "label"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    frame["title"] = frame["title"].fillna("").map(normalize_title)
    for column in ["text", "author"]:
        frame[column] = frame[column].fillna("").map(normalize_text)

    frame["body"] = frame["text"]
    frame = frame[(frame["body"].str.len() >= MIN_BODY_LENGTH) | (frame["title"].str.len() >= 15)]
    frame = frame.drop_duplicates(subset=["title", "body", "label"]).reset_index(drop=True)

    return frame[["title", "body", "author", "label"]]


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["title", "body"]].copy()
