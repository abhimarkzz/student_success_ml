from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import MODEL_DIR, REPORT_DIR, save_json
from src.performance import (
    load_uci_subject,
    split_performance,
    train_regressor,
    train_classifier,
    evaluate_regressor,
    evaluate_classifier,
)


def tune_regressor(X_fit, y_fit, X_val, y_val, cat_cols, trials: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 800, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
        model.fit(X_fit, y_fit, cat_features=cat_cols, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=100, verbose=False)
        pred = model.predict(X_val)
        return mean_squared_error(y_val, pred) ** 0.5

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=["mat", "por"], default="mat")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--tune-trials", type=int, default=20)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else ROOT / "data/raw/uci_student_performance" / f"student-{args.subject}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download UCI data first: python scripts/download_datasets.py --uci"
        )

    data = load_uci_subject(csv_path)
    X_fit, X_val, X_test, yreg_fit, yreg_val, yreg_test, ycls_fit, ycls_val, ycls_test = split_performance(data)

    best = tune_regressor(X_fit, yreg_fit, X_val, yreg_val, data.cat_cols, args.tune_trials) if args.tune_trials > 0 else {}
    reg = train_regressor(X_fit, X_val, yreg_fit, yreg_val, data.cat_cols, best)
    clf = train_classifier(X_fit, X_val, ycls_fit, ycls_val, data.cat_cols)

    reg_metrics = evaluate_regressor(reg, X_test, yreg_test)
    cls_metrics = evaluate_classifier(clf, X_test, ycls_test)

    reg_path = MODEL_DIR / f"performance_{args.subject}_regressor.cbm"
    cls_path = MODEL_DIR / f"performance_{args.subject}_pass_classifier.cbm"
    reg.save_model(reg_path)
    clf.save_model(cls_path)

    imp = pd.DataFrame({"feature": X_fit.columns, "importance": reg.get_feature_importance()}).sort_values("importance", ascending=False)
    imp.to_csv(REPORT_DIR / f"performance_{args.subject}_feature_importance.csv", index=False)

    payload = {
        "dataset": "UCI Student Performance",
        "subject": args.subject,
        "target": "G3",
        "feature_policy": "early_warning_excludes_G1_G2_G3_fail",
        "best_regressor_params": best,
        "regression_metrics": reg_metrics,
        "classification_metrics": cls_metrics,
        "model": "CatBoost",
        "note": "Metrics are measured by this run and must not be presented before training has actually produced them.",
    }
    save_json(REPORT_DIR / f"performance_{args.subject}_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
