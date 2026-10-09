"""Step 3.1: prove experiment tracking works end to end.

Run from the project root:  python scripts/tracking_smoke.py

Trains a "guess the average" baseline on the training set, scores it on the validation set,
logs everything to MLflow, then reloads the saved model and checks it predicts identically.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from mkopoguard import config
from mkopoguard.data.split import load_split
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.tracking import (
    load_pipeline,
    log_pipeline,
    setup_tracking,
    start_run,
)


def main() -> None:
    setup_tracking()
    table = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    train = load_split(table, "train")
    validation = load_split(table, "validation")
    X_train, y_train = train[input_columns()], train[TARGET]
    X_val, y_val = validation[input_columns()], validation[TARGET]

    pipeline = make_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(X_train, y_train)
    p_val = pipeline.predict_proba(X_val)[:, 1]

    with start_run("smoke-dummy-prior", evaluated_on="validation", step="3.1") as run:
        mlflow.log_params(
            {
                "model": "DummyClassifier(prior)",
                "include_country": False,
                "n_train": len(train),
                "n_validation": len(validation),
            }
        )
        mlflow.log_metrics(
            {
                "roc_auc": roc_auc_score(y_val, p_val),
                "mean_predicted_probability": float(p_val.mean()),
                "actual_default_rate": float(y_val.mean()),
            }
        )
        log_pipeline(pipeline)

    reloaded = load_pipeline(run.info.run_id)
    same = np.array_equal(reloaded.predict_proba(X_val)[:, 1], p_val)
    print(f"Logged run {run.info.run_id}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, p_val):.3f} (0.5 expected: it guesses)")
    print(f"Reloaded model predicts identically: {same}")
    print("Open the UI with:  python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")
    if not same:
        raise SystemExit("Reloaded model differs from the one that was logged")


if __name__ == "__main__":
    main()
