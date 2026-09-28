from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .common import clean_frame, coerce_catboost_categories, infer_categorical_columns


@dataclass
class PlacementBundle:
    X: pd.DataFrame
    y: pd.Series
    cat_cols: list[str]
    target: str


def load_placement_csv(csv_path: Path, target: str) -> PlacementBundle:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    df = clean_frame(df)
    y = df[target].astype(str)
    # Normalize common labels while preserving arbitrary binary labels.
    labels = {str(v).strip().lower() for v in y.unique()}
    if labels.issubset({"placed", "notplaced", "not placed"}):
        y = y.str.strip().str.lower().map({"placed": 1, "notplaced": 0, "not placed": 0}).astype(int)
    else:
        uniques = list(pd.Series(y).dropna().unique())
        if len(uniques) != 2:
            raise ValueError(f"Placement target must be binary for this implementation. Found: {uniques}")
        mapping = {uniques[0]: 0, uniques[1]: 1}
        y = y.map(mapping).astype(int)

    X = df.drop(columns=[target])
    cat_cols = infer_categorical_columns(X)
    X = coerce_catboost_categories(X, cat_cols)
    return PlacementBundle(X, y, cat_cols, target)


def train_placement_model(bundle: PlacementBundle, random_state: int = 42) -> tuple[CatBoostClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.20,
        random_state=random_state,
        stratify=bundle.y,
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=random_state,
        stratify=y_train,
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=8.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        thread_count=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        cat_features=bundle.cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=150,
        verbose=False,
    )

    pred = model.predict(X_test).astype(int).reshape(-1)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "task": "placement_classification",
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test)),
        "positive_class": 1,
    }
    return model, metrics
