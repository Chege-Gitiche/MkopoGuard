"""Step 3.8: learning curves. Would the model improve with more applicants?

For each CV fold, the model is trained on a growing share of that fold's training rows
(1/16, 1/8, 1/4, 1/2, all) and always scored on the SAME full held-out fold, so every point
on the curve is measured on identical applicants. Small shares are drawn several times with
different seeds, because a small sample can be lucky or unlucky.

Two references make the curve readable:
- the training score: the gap between it and the held-out score is overfitting, and it
  should close as data grows;
- the ceiling: the PR-AUC of the simulator's TRUE default probabilities. No model can beat
  it with any amount of data, because the outcome itself is partly random. It is read from
  the hidden simulator columns for evaluation only, never as a model input.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from mkopoguard import config
from mkopoguard.data.split import load_split
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.experiment import cv_folds
from mkopoguard.models.metrics import FAST_METRICS

pr_auc = FAST_METRICS["pr_auc"]
TRUE_PROBABILITY = "hidden_default_probability"


def subsample(rows: np.ndarray, groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A share of `rows`, keeping the same mix of `groups` (e.g. default x statement)."""
    if fraction >= 1.0:
        return rows
    picked, _ = train_test_split(
        rows, train_size=fraction, stratify=groups[rows], random_state=seed
    )
    return np.sort(picked)


def curve_points(
    train: pd.DataFrame,
    model_factory,
    drop=(),
    fractions=config.LEARNING_CURVE_FRACTIONS,
    repeats: int = config.LEARNING_CURVE_REPEATS,
    seed: int = config.SPLIT_SEED,
) -> pd.DataFrame:
    """One row per (fraction, fold, repeat): training size and PR-AUC scores."""
    columns = input_columns(drop=drop)
    pipeline = make_pipeline(model_factory(), drop=drop)
    X, y = train[columns], train[TARGET].to_numpy()
    thin = (train["has_statement"] == 0).to_numpy()
    groups = (train[TARGET].astype(str) + "|" + train["has_statement"].astype(str)).to_numpy()
    rows = []
    for fold, (fit_rows, held_out) in enumerate(cv_folds(train)):
        y_held, thin_held = y[held_out], thin[held_out]
        for fraction in fractions:
            for repeat in range(repeats if fraction < 1.0 else 1):
                used = subsample(fit_rows, groups, fraction, seed + repeat)
                fitted = clone(pipeline).fit(X.iloc[used], y[used])
                p = fitted.predict_proba(X.iloc[held_out])[:, 1]
                p_fit = fitted.predict_proba(X.iloc[used])[:, 1]
                rows.append(
                    {
                        "fraction": fraction,
                        "fold": fold,
                        "repeat": repeat,
                        "n_train": len(used),
                        "cv_pr_auc": pr_auc(y_held, p),
                        "cv_pr_auc_thin_file": pr_auc(y_held[thin_held], p[thin_held]),
                        "train_pr_auc": pr_auc(y[used], p_fit),
                    }
                )
    return pd.DataFrame(rows)


def summarise(points: pd.DataFrame) -> pd.DataFrame:
    """Average the points per fraction; the std shows how much the score moves between runs."""
    table = points.groupby("fraction").agg(
        n_train=("n_train", "mean"),
        cv_pr_auc=("cv_pr_auc", "mean"),
        cv_pr_auc_std=("cv_pr_auc", "std"),
        cv_pr_auc_thin_file=("cv_pr_auc_thin_file", "mean"),
        train_pr_auc=("train_pr_auc", "mean"),
    )
    table["n_train"] = table["n_train"].round().astype(int)
    table["overfit_gap"] = table["train_pr_auc"] - table["cv_pr_auc"]
    return table


def projected_gain(summary: pd.DataFrame, column: str = "cv_pr_auc") -> dict[str, float]:
    """Gain from the last two doublings of data, and a projection for the next one.

    Gains from each doubling usually shrink. The next doubling is projected to shrink by the
    same ratio as the last one did (capped at 1, so a lucky jump can't project growth).
    """
    scores = summary[column].to_numpy()
    last, previous = scores[-1] - scores[-2], scores[-2] - scores[-3]
    if last <= 0:
        return {"last_gain": last, "previous_gain": previous, "next_gain": 0.0}
    ratio = min(last / previous, 1.0) if previous > 0 else 1.0
    return {"last_gain": last, "previous_gain": previous, "next_gain": last * ratio}


def more_data_helps(gain: dict[str, float], min_gain=config.LEARNING_CURVE_MIN_GAIN) -> bool:
    """Is another doubling of the applicants projected to gain a meaningful amount?"""
    return gain["next_gain"] > min_gain


def true_probability(train: pd.DataFrame) -> np.ndarray:
    """The simulator's true default probability for each training applicant (evaluation only).

    Read through load_split, so the locked test set can't be touched even here.
    """
    path = config.DATA_PROCESSED / "applicants.parquet"
    applicants = pd.read_parquet(path, columns=["applicant_id", TRUE_PROBABILITY])
    known = load_split(applicants, "train").set_index("applicant_id")[TRUE_PROBABILITY]
    values = train["applicant_id"].map(known)
    if values.isna().any():
        raise ValueError("Some applicants have no true probability; is this the training set?")
    return values.to_numpy()


def ceiling(train: pd.DataFrame, p_true: np.ndarray) -> dict[str, float]:
    """PR-AUC of the true probabilities on the same held-out folds: the best any model can do."""
    y = train[TARGET].to_numpy()
    thin = (train["has_statement"] == 0).to_numpy()
    scores = [
        (pr_auc(y[h], p_true[h]), pr_auc(y[h][thin[h]], p_true[h][thin[h]]))
        for _, h in cv_folds(train)
    ]
    return {
        "all": float(np.mean([s[0] for s in scores])),
        "thin_file": float(np.mean([s[1] for s in scores])),
    }
