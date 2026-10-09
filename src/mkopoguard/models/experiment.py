"""Step 3.3: one harness for every model experiment in Phase 3.

run_experiment() does the same five things for every model, so results are comparable:
1. 5-fold cross-validation on the TRAINING set only (the evaluation plan's deciding score)
2. fit on the whole training set, and record its training score to spot overfitting
3. score the VALIDATION set, overall and for Kenya, statement holders and thin-file applicants
4. stop if the scores trip the leakage alarm
5. log parameters, metrics, tables and the fitted pipeline to MLflow

The test set is never touched here.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from mkopoguard import config
from mkopoguard.data.split import STRATIFY, load_split
from mkopoguard.features.columns import TARGET, assert_no_leakage
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.metrics import assert_no_leakage_alarm, calibration_table, evaluate
from mkopoguard.tracking import log_pipeline, start_run

HEADLINE = ["pr_auc", "roc_auc", "ks", "brier", "ece", "recall", "approval_rate"]


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training and validation sets. The test set stays locked."""
    table = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    return load_split(table, "train"), load_split(table, "validation")


def cv_folds(train: pd.DataFrame, n_splits: int = config.CV_FOLDS, seed: int = config.SPLIT_SEED):
    """Stratified folds using the same groups as the split (country x default x statement)."""
    groups = train[STRATIFY].astype(str).agg("|".join, axis=1)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(folds.split(train, groups))


def subgroups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The groups every result is reported for (evaluation plan, rule 4)."""
    return {
        "all": np.ones(len(frame), dtype=bool),
        "kenya": (frame["country_code"] == config.TARGET_COUNTRY).to_numpy(),
        "statement": (frame["has_statement"] == 1).to_numpy(),
        "thin_file": (frame["has_statement"] == 0).to_numpy(),
    }


def evaluate_groups(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """evaluate() per subgroup; a group without both outcomes gets a row of NaN."""
    rows = {}
    y = frame[TARGET].to_numpy()
    for name, mask in subgroups(frame).items():
        if len(np.unique(y[mask])) == 2:
            rows[name] = evaluate(y[mask], p[mask])
        else:
            rows[name] = {"n": float(mask.sum())}
    return pd.DataFrame(rows).T


def cross_validate(pipeline, train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """evaluate() on each held-out fold. The pipeline is re-fitted from scratch per fold."""
    X, y = train[columns], train[TARGET]
    rows = []
    for train_idx, held_idx in cv_folds(train):
        fold_model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        p = fold_model.predict_proba(X.iloc[held_idx])[:, 1]
        rows.append(evaluate(y.iloc[held_idx], p))
    return pd.DataFrame(rows)


def run_experiment(
    name: str,
    model,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    step: str,
    params: dict | None = None,
    include_country: bool = False,
) -> dict:
    """Cross-validate, fit, validate and log one model. Returns the headline results."""
    columns = input_columns(include_country)
    assert_no_leakage(columns)
    pipeline = make_pipeline(model, include_country)

    with start_run(name, evaluated_on="validation", step=step) as run:
        start = time.time()
        cv = cross_validate(pipeline, train, columns)
        pipeline.fit(train[columns], train[TARGET])
        on_train = evaluate(train[TARGET], pipeline.predict_proba(train[columns])[:, 1])
        p_val = pipeline.predict_proba(validation[columns])[:, 1]
        by_group = evaluate_groups(validation, p_val)
        overall = by_group.loc["all"].to_dict()
        assert_no_leakage_alarm(overall)  # stops here (run marked FAILED) if too good

        mlflow.log_params(
            {
                "model": type(model).__name__,
                "include_country": include_country,
                "n_features_in": len(columns),
                "cv_folds": config.CV_FOLDS,
                **(params or {}),
            }
        )
        mlflow.log_metrics({f"cv_{k}_mean": cv[k].mean() for k in HEADLINE})
        mlflow.log_metrics({f"cv_{k}_std": cv[k].std() for k in HEADLINE})
        for group, row in by_group.iterrows():
            prefix = "val_" if group == "all" else f"val_{group}_"
            mlflow.log_metrics({f"{prefix}{k}": row[k] for k in HEADLINE if k in row})
        # A big gap between training and validation scores means the model memorised noise
        mlflow.log_metrics(
            {
                "train_pr_auc": on_train["pr_auc"],
                "train_roc_auc": on_train["roc_auc"],
                "overfit_gap_pr_auc": on_train["pr_auc"] - overall["pr_auc"],
                "fit_seconds": time.time() - start,
            }
        )
        mlflow.log_text(cv.to_csv(index=False), "cv_folds.csv")
        mlflow.log_text(by_group.to_csv(), "validation_by_group.csv")
        mlflow.log_text(
            calibration_table(validation[TARGET], p_val).to_csv(index=False),
            "validation_calibration.csv",
        )
        log_pipeline(pipeline)

    return {
        "name": name,
        "run_id": run.info.run_id,
        "cv_pr_auc": cv["pr_auc"].mean(),
        "cv_pr_auc_std": cv["pr_auc"].std(),
        "train_pr_auc": on_train["pr_auc"],
        "overfit_gap": on_train["pr_auc"] - overall["pr_auc"],
        **{f"val_{k}": overall[k] for k in HEADLINE},
        "val_kenya_pr_auc": by_group.loc["kenya"].get("pr_auc", np.nan),
        "val_thin_file_pr_auc": by_group.loc["thin_file"].get("pr_auc", np.nan),
        "pipeline": pipeline,
        "p_val": p_val,
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    """Side-by-side comparison, best cross-validated PR-AUC first."""
    keep = [k for k in results[0] if k not in ("pipeline", "p_val", "run_id")]
    table = pd.DataFrame([{k: r[k] for k in keep} for r in results]).set_index("name")
    return table.sort_values("cv_pr_auc", ascending=False)
