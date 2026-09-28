from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .common import clean_frame, coerce_catboost_categories

UCI_TARGET = "G3"
LEAKY_EARLY_FEATURES = {"G1", "G2", "G3", "fail"}


@dataclass
class PerformanceDataset:
    X: pd.DataFrame
    y_reg: pd.Series
    y_cls: pd.Series
    cat_cols: list[str]


def load_uci_subject(csv_path: Path) -> PerformanceDataset:
    df = pd.read_csv(csv_path, sep=";")
    if UCI_TARGET not in df.columns:
        raise ValueError(f"Expected {UCI_TARGET} in {csv_path}")

    # Early-warning setting: remove G1/G2/G3/fail to avoid temporal leakage or
    # target-derived leakage. G3 remains the target only.
    feature_cols = [c for c in df.columns if c not in LEAKY_EARLY_FEATURES]
    X = clean_frame(df[feature_cols])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    X = coerce_catboost_categories(X, cat_cols)

    y_reg = pd.to_numeric(df[UCI_TARGET], errors="raise").astype(float)
    y_cls = (y_reg >= 10.0).astype(int)
    return PerformanceDataset(X, y_reg, y_cls, cat_cols)


def split_performance(data: PerformanceDataset, random_state: int = 42):
    X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
        data.X,
        data.y_reg,
        data.y_cls,
        test_size=0.20,
        random_state=random_state,
        stratify=data.y_cls,
    )
    X_fit, X_val, yreg_fit, yreg_val, ycls_fit, ycls_val = train_test_split(
        X_train,
        yreg_train,
        ycls_train,
        test_size=0.20,
        random_state=random_state,
        stratify=ycls_train,
    )
    return X_fit, X_val, X_test, yreg_fit, yreg_val, yreg_test, ycls_fit, ycls_val, ycls_test


def train_regressor(X_fit, X_val, y_fit, y_val, cat_cols, params: dict | None = None) -> CatBoostRegressor:
    defaults = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    if params:
        defaults.update(params)
    model = CatBoostRegressor(**defaults)
    model.fit(
        X_fit,
        y_fit,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=150,
        verbose=False,
    )
    return model


def train_classifier(X_fit, X_val, y_fit, y_val, cat_cols, params: dict | None = None) -> CatBoostClassifier:
    defaults = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        auto_class_weights="Balanced",
    )
    if params:
        defaults.update(params)
    model = CatBoostClassifier(**defaults)
    model.fit(
        X_fit,
        y_fit,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=150,
        verbose=False,
    )
    return model


def evaluate_regressor(model, X_test, y_test) -> dict:
    pred = np.asarray(model.predict(X_test)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "task": "final_grade_regression",
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, pred)),
        "n_test": int(len(y_test)),
    }


def evaluate_classifier(model, X_test, y_test) -> dict:
    pred = model.predict(X_test).astype(int).reshape(-1)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "task": "pass_risk_classification",
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test)),
    }
