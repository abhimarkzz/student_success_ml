from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
EXAMPLE_DIR = ROOT / "examples"

for d in (RAW_DIR, MODEL_DIR, REPORT_DIR, EXAMPLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).fillna("Unknown")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def infer_categorical_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [c for c in df.columns if c not in excluded and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]


def coerce_catboost_categories(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].fillna("Unknown").astype(str)
    return out


def sanitize_for_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
