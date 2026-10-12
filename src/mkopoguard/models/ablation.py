"""Step 3.7: feature ablation. Which groups of inputs does the model actually need?

Method: remove one group at a time, re-run the same 5-fold CV, and compare each fold with
the same fold of the full model (a paired comparison: same applicants, so the fold-to-fold
noise mostly cancels out).

Rule 7 (evaluation plan): a group is NEEDED if removing it causes a loss that is both
- consistent: the average paired loss is bigger than its standard error, and
- big enough to matter: bigger than config.ABLATION_MIN_LOSS,
for ALL applicants or for THIN-FILE applicants. Thin-file applicants are checked on their own
because they have no statement, so the survey answers are all the model knows about them;
a group can look useless overall and still matter for them.
Groups that aren't needed are dropped together, and the combined drop is checked the same
way (on CV) and then on the validation set before it is adopted.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from mkopoguard import config
from mkopoguard.features.columns import TARGET
from mkopoguard.features.groups import FEATURE_GROUPS, columns_in
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.experiment import cv_folds
from mkopoguard.models.metrics import FAST_METRICS

pr_auc = FAST_METRICS["pr_auc"]
FOCUS = ("all", "thin_file")  # the applicant groups rule 7 checks


def fold_scores(train: pd.DataFrame, model, drop_groups=()) -> pd.DataFrame:
    """PR-AUC on each held-out CV fold, for all applicants and for thin-file applicants."""
    drop = columns_in(drop_groups)
    columns = input_columns(drop=drop)
    pipeline = make_pipeline(model, drop=drop)
    X, y = train[columns], train[TARGET].to_numpy()
    thin = (train["has_statement"] == 0).to_numpy()
    rows = []
    for fit_rows, held_out in cv_folds(train):
        fitted = clone(pipeline).fit(X.iloc[fit_rows], y[fit_rows])
        p = fitted.predict_proba(X.iloc[held_out])[:, 1]
        y_held, thin_held = y[held_out], thin[held_out]
        rows.append(
            {"all": pr_auc(y_held, p), "thin_file": pr_auc(y_held[thin_held], p[thin_held])}
        )
    return pd.DataFrame(rows)


def paired_loss(base: pd.DataFrame, other: pd.DataFrame) -> dict[str, float]:
    """How much worse `other` is than `base`, fold by fold. Positive = other is worse."""
    out = {}
    for focus in FOCUS:
        diff = base[focus].to_numpy() - other[focus].to_numpy()
        out[f"loss_{focus}"] = float(diff.mean())
        out[f"se_{focus}"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        out[f"folds_worse_{focus}"] = int((diff > 0).sum())
    return out


def is_needed(loss: dict[str, float], min_loss: float = config.ABLATION_MIN_LOSS) -> bool:
    """Rule 7: a consistent AND meaningful loss, for all applicants or thin-file applicants."""
    return any(loss[f"loss_{f}"] > loss[f"se_{f}"] and loss[f"loss_{f}"] > min_loss for f in FOCUS)


def drop_one_table(train: pd.DataFrame, model_factory, full: pd.DataFrame) -> pd.DataFrame:
    """For each group: CV without it, CV with only it, the paired loss, and the verdict."""
    rows = []
    for group in FEATURE_GROUPS:
        others = [g for g in FEATURE_GROUPS if g != group]
        without = fold_scores(train, model_factory(), drop_groups=[group])
        only = fold_scores(train, model_factory(), drop_groups=others)
        loss = paired_loss(full, without)
        rows.append(
            {
                "group": group,
                "n_columns": len(FEATURE_GROUPS[group]),
                "cv_pr_auc_without": without["all"].mean(),
                "cv_pr_auc_only": only["all"].mean(),
                **loss,
                "needed": is_needed(loss),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def paired_bootstrap(
    y, p_base, p_other, resamples: int = config.BOOTSTRAP_RESAMPLES, seed: int = 0
) -> dict[str, float]:
    """PR-AUC loss (base - other) on the same applicants, with a 95% interval.

    Both models are scored on the SAME resampled applicants each time, so the interval
    describes the difference itself, not two separate scores.
    """
    y, p_base, p_other = np.asarray(y), np.asarray(p_base), np.asarray(p_other)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(pr_auc(y[i], p_base[i]) - pr_auc(y[i], p_other[i]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"loss": pr_auc(y, p_base) - pr_auc(y, p_other), "low": low, "high": high}


def save_selection(dropped: list[str], path: Path = config.FEATURE_SELECTION_FILE) -> Path:
    """Record which groups later steps leave out (and so which questions the form skips)."""
    kept = [g for g in FEATURE_GROUPS if g not in dropped]
    selection = {"dropped_groups": list(dropped), "kept_groups": kept}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return path


def chosen_drop(path: Path = config.FEATURE_SELECTION_FILE) -> list[str]:
    """The columns later steps (3.8 onwards) pass as make_pipeline(..., drop=...)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/ablate_features.py first")
    selection = json.loads(path.read_text(encoding="utf-8"))
    return columns_in(selection["dropped_groups"])
