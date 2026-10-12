"""Step 3.9: leave-one-country-out. Does the model work in a country it has never seen?

A lender may take MkopoGuard to a new market. Country was dropped as an input in step 3.3,
but the model could still rely on patterns that only hold in the countries it learned from.

Design (paired, so only one thing changes):
for each CV fold, and for each country C, score the held-out applicants from C twice:
- SEEN:   the model trained on the fold's training rows (which include other applicants from C)
- UNSEEN: the same model trained on the same rows MINUS every applicant from C
Same applicants, same fold, ~5% fewer training rows; the only real difference is whether
the model has ever seen country C.

Rule 9 (evaluation plan): the model transfers to new countries if
- no country loses clearly: a loss bigger than config.COUNTRY_MIN_LOSS whose paired bootstrap
  interval lies entirely above zero (the interval is widened for checking 20 countries), and
- no country's average prediction is off by more than chance (calibration-in-the-large,
  Bonferroni-adjusted, as in step 3.10) when the model has never seen it.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from mkopoguard import config
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.ablation import paired_bootstrap
from mkopoguard.models.calibration import group_calibration
from mkopoguard.models.experiment import cv_folds
from mkopoguard.models.metrics import FAST_METRICS

COUNTRY = "country_code"
pr_auc, roc_auc = FAST_METRICS["pr_auc"], FAST_METRICS["roc_auc"]


def seen_and_unseen(train: pd.DataFrame, model_factory, drop=()) -> pd.DataFrame:
    """Score every training applicant with, and without, their country in the training data."""
    columns = input_columns(drop=drop)
    pipeline = make_pipeline(model_factory(), drop=drop)
    X, y = train[columns], train[TARGET].to_numpy()
    country = train[COUNTRY].to_numpy()
    seen, unseen = np.full(len(train), np.nan), np.full(len(train), np.nan)
    for fit_rows, held_out in cv_folds(train):
        model = clone(pipeline).fit(X.iloc[fit_rows], y[fit_rows])
        seen[held_out] = model.predict_proba(X.iloc[held_out])[:, 1]
        for code in np.unique(country[held_out]):
            target = held_out[country[held_out] == code]
            others = fit_rows[country[fit_rows] != code]
            without = clone(pipeline).fit(X.iloc[others], y[others])
            unseen[target] = without.predict_proba(X.iloc[target])[:, 1]
    return pd.DataFrame(
        {COUNTRY: country, TARGET: y, "p_seen": seen, "p_unseen": unseen}, index=train.index
    )


def country_table(
    preds: pd.DataFrame,
    min_loss: float = config.COUNTRY_MIN_LOSS,
    resamples: int = config.BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    """Per country: ranking with and without having seen it, and whether it loses clearly."""
    codes = sorted(preds[COUNTRY].unique())
    level = 1 - 0.05 / len(codes)  # widen the interval for checking many countries at once
    rows = []
    for code in codes:
        part = preds[preds[COUNTRY] == code]
        y, seen, unseen = part[TARGET].to_numpy(), part["p_seen"], part["p_unseen"]
        loss = paired_bootstrap(y, seen, unseen, resamples=resamples, level=level)
        rows.append(
            {
                COUNTRY: code,
                "n": len(part),
                "default_rate": y.mean(),
                "pr_auc_seen": pr_auc(y, seen),
                "pr_auc_unseen": pr_auc(y, unseen),
                "loss": loss["loss"],
                "loss_low": loss["low"],
                "loss_high": loss["high"],
                "roc_auc_seen": roc_auc(y, seen),
                "roc_auc_unseen": roc_auc(y, unseen),
                "clear_loss": bool(loss["low"] > 0 and loss["loss"] > min_loss),
            }
        )
    return pd.DataFrame(rows).set_index(COUNTRY)


def unseen_calibration(preds: pd.DataFrame) -> pd.DataFrame:
    """Calibration-in-the-large per country, for the model that has never seen it."""
    return group_calibration(preds, preds["p_unseen"].to_numpy(), COUNTRY)


def transfers(table: pd.DataFrame, calibration: pd.DataFrame) -> bool:
    """Rule 9: no country loses clearly, and none is miscalibrated beyond chance."""
    return not table["clear_loss"].any() and not calibration["flagged"].any()
