"""Step 3.10: are the predicted probabilities honest, and does recalibrating help?

A loan officer will read "23% chance of default" literally, and step 3.11 sets the decision
threshold from costs, which only works if 23% really means about 23 in 100 default.

Measured on out-of-fold CV predictions (every training applicant is scored by a model that
never saw them), with:
- ECE: average gap between predicted and actual default rates across 10 probability bins
- Brier score: average squared error of the probabilities
- calibration slope and intercept: refit y ~ a + b * logit(p). Perfect is b = 1, a = 0;
  b < 1 means over-confident (too extreme), b > 1 under-confident
- calibration-in-the-large per group (each country, thin-file vs statement): does the
  average prediction match the actual default rate?

Methods compared: none, sigmoid (Platt scaling) and isotonic regression, both fitted with
an inner 5-fold CV inside each training fold, so the calibrator never sees held-out rows.

Rule 8 (evaluation plan): a method is adopted only if, compared with none, it cuts ECE by
more than its paired standard error AND by more than config.CALIBRATION_MIN_ECE_GAIN,
without making the Brier score worse or costing a meaningful amount of PR-AUC.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from mkopoguard import config
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.experiment import cv_folds
from mkopoguard.models.metrics import FAST_METRICS, expected_calibration_error

pr_auc, brier = FAST_METRICS["pr_auc"], FAST_METRICS["brier"]


def calibrated_model(model, method: str = "none"):
    """The model, optionally wrapped so its probabilities are recalibrated.

    The wrapper goes INSIDE the pipeline (after preprocessing) and learns the correction
    from its own inner 5-fold CV on the training rows it is given.
    """
    if method == "none":
        return model
    if method not in config.CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}")
    return CalibratedClassifierCV(model, method=method, cv=5)


def out_of_fold(train: pd.DataFrame, model, drop=()) -> np.ndarray:
    """A prediction for every training applicant from the CV fold that held them out."""
    columns = input_columns(drop=drop)
    pipeline = make_pipeline(model, drop=drop)
    X, y = train[columns], train[TARGET].to_numpy()
    p = np.full(len(train), np.nan)
    for fit_rows, held_out in cv_folds(train):
        fitted = clone(pipeline).fit(X.iloc[fit_rows], y[fit_rows])
        p[held_out] = fitted.predict_proba(X.iloc[held_out])[:, 1]
    return p


def calibration_slope(y, p) -> tuple[float, float]:
    """Slope and intercept of y ~ a + b * logit(p). Perfect calibration: (1, 0)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, np.asarray(y))
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def summary(y, p) -> dict[str, float]:
    """Calibration and ranking numbers for one set of predictions."""
    slope, intercept = calibration_slope(y, p)
    return {
        "pr_auc": float(pr_auc(y, p)),
        "brier": float(brier(y, p)),
        "ece": expected_calibration_error(y, p),
        "slope": slope,
        "intercept": intercept,
        "mean_predicted": float(np.mean(p)),
        "default_rate": float(np.mean(y)),
    }


def fold_metrics(train: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """ECE, Brier and PR-AUC on each held-out fold, for paired comparisons."""
    y = train[TARGET].to_numpy()
    return pd.DataFrame(
        [
            {
                "ece": expected_calibration_error(y[h], p[h]),
                "brier": brier(y[h], p[h]),
                "pr_auc": pr_auc(y[h], p[h]),
            }
            for _, h in cv_folds(train)
        ]
    )


def calibration_gain(base: pd.DataFrame, other: pd.DataFrame) -> dict[str, float]:
    """How much better `other` is than `base`, fold by fold (positive = better)."""

    def paired(column: str, sign: int):
        diff = sign * (base[column].to_numpy() - other[column].to_numpy())
        return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))

    ece_gain, ece_se = paired("ece", +1)  # lower ECE is better
    brier_gain, brier_se = paired("brier", +1)  # lower Brier is better
    pr_gain, _ = paired("pr_auc", -1)  # higher PR-AUC is better
    return {
        "ece_gain": ece_gain,
        "ece_gain_se": ece_se,
        "brier_gain": brier_gain,
        "brier_gain_se": brier_se,
        "pr_auc_gain": pr_gain,
    }


def worth_calibrating(
    gain: dict[str, float],
    min_ece_gain: float = config.CALIBRATION_MIN_ECE_GAIN,
    max_pr_auc_loss: float = config.ABLATION_MIN_LOSS,
) -> bool:
    """Rule 8: a clear, meaningful ECE cut, no worse Brier, no meaningful PR-AUC loss."""
    clear = gain["ece_gain"] > gain["ece_gain_se"] and gain["ece_gain"] > min_ece_gain
    brier_ok = gain["brier_gain"] > -gain["brier_gain_se"]
    ranking_ok = gain["pr_auc_gain"] > -max_pr_auc_loss
    return clear and brier_ok and ranking_ok


def flag_z(n_groups: int, alpha: float = 0.05) -> float:
    """How far (in standard errors) a group must be off before it is flagged.

    With many groups, some will be off by chance (1 in 20 at the usual 2 standard errors),
    so the bar is raised for the number of groups checked (Bonferroni correction).
    """
    return float(norm.ppf(1 - alpha / (2 * n_groups)))


def group_calibration(frame: pd.DataFrame, p: np.ndarray, by: str) -> pd.DataFrame:
    """Calibration-in-the-large per group: observed vs expected default rate.

    The standard error assumes the predictions are right, so z says how surprising the
    observed rate would be if they were.
    """
    data = pd.DataFrame({"group": frame[by].to_numpy(), "y": frame[TARGET].to_numpy(), "p": p})
    data["var"] = data["p"] * (1 - data["p"])
    table = data.groupby("group").agg(
        n=("y", "size"), observed=("y", "mean"), expected=("p", "mean"), var=("var", "sum")
    )
    table["se"] = np.sqrt(table.pop("var")) / table["n"]
    table["observed_to_expected"] = table["observed"] / table["expected"]
    table["z"] = (table["observed"] - table["expected"]) / table["se"]
    table["flagged"] = table["z"].abs() > flag_z(len(table))
    return table


def save_choice(method: str, path: Path = config.CALIBRATION_FILE) -> Path:
    """Record the calibration method later steps use."""
    if method not in config.CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": method}, indent=2) + "\n", encoding="utf-8")
    return path


def chosen_calibration(path: Path = config.CALIBRATION_FILE) -> str:
    """The method step 3.10 chose; later steps build calibrated_model(model, this)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/calibrate.py first")
    return json.loads(path.read_text(encoding="utf-8"))["method"]
