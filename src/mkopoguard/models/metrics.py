"""Step 3.2: how every model is judged, decided BEFORE any model is trained.

evaluate() returns the same set of numbers for every model, so models are always compared
like for like. The plan behind these choices is written in docs/evaluation_plan.md.

Ranking:        pr_auc (PRIMARY), roc_auc, ks
Probabilities:  brier, brier_skill, ece (expected calibration error)
Decisions:      recall, precision, false_positive_rate, accuracy, approval_rate
                (at a threshold; PROVISIONAL_THRESHOLD until step 3.11 sets the real one)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from mkopoguard import config

CALIBRATION_BINS = 10


def ks_statistic(y_true, p) -> float:
    """Largest gap between the score distributions of defaulters and non-defaulters.

    Equal to the maximum of (true positive rate - false positive rate) along the ROC curve:
    0 = the model can't separate the groups, 1 = it separates them perfectly.
    """
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def calibration_table(y_true, p, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Predicted vs actual default rate in equal-width probability bins (0-0.1, 0.1-0.2, ...).

    Empty bins are left out. A well-calibrated model has predicted close to actual in every row.
    """
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bin": which, "predicted": p, "actual": y_true})
    table = frame.groupby("bin").agg(
        applicants=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean")
    )
    table.insert(0, "upper", edges[1:][table.index])
    table.insert(0, "lower", edges[:-1][table.index])
    return table.reset_index(drop=True)


def expected_calibration_error(y_true, p, bins: int = CALIBRATION_BINS) -> float:
    """Average |predicted - actual| across bins, weighted by how many applicants are in each."""
    table = calibration_table(y_true, p, bins)
    weights = table["applicants"] / table["applicants"].sum()
    return float((weights * (table["predicted"] - table["actual"]).abs()).sum())


def decision_metrics(y_true, p, threshold: float) -> dict[str, float]:
    """Metrics for the yes/no decision: decline when p >= threshold."""
    y_true = np.asarray(y_true).astype(int)
    declined = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, declined, labels=[0, 1]).ravel()
    return {
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "accuracy": (tp + tn) / len(y_true),
        "approval_rate": 1 - declined.mean(),
    }


def evaluate(y_true, p, threshold: float = config.PROVISIONAL_THRESHOLD) -> dict[str, float]:
    """Every metric in the evaluation plan, for one set of predictions.

    y_true: 1 = defaulted. p: predicted probability of default.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both defaulters and non-defaulters to evaluate ranking metrics")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Predicted probabilities must all be between 0 and 1")

    base_rate = y_true.mean()
    brier = brier_score_loss(y_true, p)
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "ks": ks_statistic(y_true, p),
        "brier": float(brier),
        "brier_skill": float(1 - brier / (base_rate * (1 - base_rate))),
        "ece": expected_calibration_error(y_true, p),
        **{k: float(v) for k, v in decision_metrics(y_true, p, threshold).items()},
        "threshold": float(threshold),
        "default_rate": float(base_rate),
        "mean_predicted": float(p.mean()),
        "n": float(len(y_true)),
    }


# Metrics that can be computed on their own, so bootstrapping doesn't recompute everything
FAST_METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "ks": ks_statistic,
    "brier": brier_score_loss,
}
DECISION_METRICS = ("recall", "precision", "false_positive_rate", "accuracy", "approval_rate")


def bootstrap_ci(
    y_true,
    p,
    metric: str = "roc_auc",
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
    threshold: float = config.PROVISIONAL_THRESHOLD,
) -> tuple[float, float]:
    """Confidence interval for one metric, by resampling applicants with replacement.

    Used for small groups such as the ~144 Kenyans in the test set.
    """
    y_true, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    if metric in FAST_METRICS:
        score = FAST_METRICS[metric]
    elif metric in DECISION_METRICS:
        score = lambda y, q: decision_metrics(y, q, threshold)[metric]  # noqa: E731
    else:
        score = lambda y, q: evaluate(y, q, threshold)[metric]  # noqa: E731
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:  # a resample needs both outcomes
            scores.append(score(y_true[idx], p[idx]))
    alpha = (1 - level) / 2
    low, high = np.quantile(scores, [alpha, 1 - alpha])
    return float(low), float(high)


def evaluate_by_group(y_true, p, groups: dict, threshold=config.PROVISIONAL_THRESHOLD):
    """One row of metrics per named group, e.g. {'all': mask, 'kenya': mask, ...}."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    rows = {name: evaluate(y_true[mask], p[mask], threshold) for name, mask in groups.items()}
    return pd.DataFrame(rows).T


def leakage_alarms(metrics: dict) -> dict[str, str]:
    """Scores above what is possible with the TRUE default probabilities mean a leak.

    The ceiling (step 3.2) is ROC-AUC 0.845 and PR-AUC 0.615 on every split.
    """
    return {
        name: f"{metrics[name]:.3f} exceeds the ceiling alarm of {limit}"
        for name, limit in config.LEAKAGE_ALARM.items()
        if metrics[name] > limit
    }


def assert_no_leakage_alarm(metrics: dict) -> None:
    alarms = leakage_alarms(metrics)
    if alarms:
        details = "; ".join(f"{k}: {v}" for k, v in alarms.items())
        raise ValueError(f"Suspiciously good: {details}. Check for leakage before trusting it.")
