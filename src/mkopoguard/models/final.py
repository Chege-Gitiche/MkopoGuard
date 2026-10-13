"""Step 3.12: score the frozen champion on the locked test set, ONCE.

Everything about the champion was decided on training and validation data (3.3-3.11):
the model and its settings, the feature groups, calibration and the decision rule. The test
set has never been looked at. Its job is to give an honest final number, so:

- the decisions are frozen first: their files are fingerprinted into the report, so the
  report proves which model it describes;
- the code must be committed (a "-dirty" Git commit is refused), so the result can be
  traced to exact code;
- the test can be opened only once. A second run needs a written reason, which is logged
  in docs/results/test_runs.log for anyone reviewing the project to see. Nothing about the
  model may be changed because of what the test shows.

The final model is refitted on training + validation applicants with the frozen settings
(the learning curve in 3.8 showed this changes little, but it is the model that ships).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.data.split import load_split
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import make_pipeline
from mkopoguard.models import decision as dec
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.calibration import calibrated_model, chosen_calibration, summary
from mkopoguard.models.experiment import subgroups
from mkopoguard.models.metrics import FAST_METRICS, bootstrap_ci, expected_calibration_error
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import file_fingerprint

# The decisions that define the champion. The test report records a fingerprint of each.
FROZEN_FILES = {
    "tuned_params": config.TUNED_PARAMS_FILE,
    "feature_selection": config.FEATURE_SELECTION_FILE,
    "calibration": config.CALIBRATION_FILE,
    "decision_policy": config.DECISION_POLICY_FILE,
}
RANKING = ("pr_auc", "roc_auc", "ks")


def frozen_decisions(files: dict[str, Path] = FROZEN_FILES) -> dict[str, str]:
    """Fingerprint of every decision file; fails if any decision hasn't been made yet."""
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Decisions not made yet: {missing}. Finish steps 3.6-3.11.")
    return {name: file_fingerprint(path) for name, path in files.items()}


def final_pipeline():
    """The champion exactly as steps 3.6, 3.7 and 3.10 decided (unfitted)."""
    model = calibrated_model(chosen_model("logistic"), chosen_calibration())
    return make_pipeline(model, drop=chosen_drop())


def check_can_open(
    commit: str,
    reason: str | None = None,
    report: Path = config.TEST_REPORT_FILE,
) -> None:
    """Refuse to open the test set with uncommitted code, or a second time without a reason."""
    if commit.endswith("-dirty"):
        raise PermissionError(
            "Uncommitted changes: commit your work first, so the test result is tied to exact code."
        )
    if report.exists() and not reason:
        raise PermissionError(
            f"The test set was already opened (see {report.name}). Opening it again needs a "
            "written reason (--rerun-reason), and the model must not be changed because of it."
        )


def log_opening(commit: str, reason: str | None, log: Path = config.TEST_RUNS_LOG) -> None:
    """Append one line per opening of the test set: when, which commit, and why."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    why = reason or "first and planned opening (step 3.12)"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} | commit {commit} | {why}\n")


def with_ci(y, p, metric: str, resamples: int = config.BOOTSTRAP_RESAMPLES) -> dict:
    low, high = bootstrap_ci(y, p, metric, resamples=resamples)
    return {"value": float(FAST_METRICS[metric](y, p)), "low": low, "high": high}


def group_report(frame: pd.DataFrame, p: np.ndarray, resamples=config.BOOTSTRAP_RESAMPLES):
    """Per subgroup: size, default rate, PR-AUC with 95% CI, calibration and approvals."""
    y = frame[TARGET].to_numpy()
    approve = dec.cost_rule(p, frame["term_months"])
    rows = {}
    for name, mask in subgroups(frame).items():
        pr = with_ci(y[mask], p[mask], "pr_auc", resamples)
        rows[name] = {
            "n": int(mask.sum()),
            "default_rate": float(y[mask].mean()),
            "mean_predicted": float(p[mask].mean()),
            "pr_auc": pr["value"],
            "pr_auc_low": pr["low"],
            "pr_auc_high": pr["high"],
            "ece": expected_calibration_error(y[mask], p[mask]),
            "approval_rate": float(approve[mask].mean()),
            "default_rate_approved": float(y[mask & approve].mean()),
        }
    return pd.DataFrame(rows).T


def build_report(frame: pd.DataFrame, p: np.ndarray, resamples=config.BOOTSTRAP_RESAMPLES):
    """Everything the evaluation plan promised, on one set of predictions."""
    y = frame[TARGET].to_numpy()
    amount, term = frame["loan_amount_kes"], frame["term_months"]
    approve = dec.cost_rule(p, term)
    everyone = np.ones(len(p), dtype=bool)
    return {
        "n": len(frame),
        "default_rate": float(y.mean()),
        "ranking": {m: with_ci(y, p, m, resamples) for m in RANKING},
        "calibration": summary(y, p),
        "decision": {
            "cost_rule": dec.policy_summary("cost_rule", y, approve, amount, term),
            "approve_everyone": dec.policy_summary("approve_everyone", y, everyone, amount, term),
            "gain_vs_approve_everyone_per_1000": dec.profit_difference(
                y, approve, everyone, amount, term, resamples
            ),
        },
        "groups": group_report(frame, p, resamples).to_dict(orient="index"),
    }


def true_probability_on_test(test: pd.DataFrame) -> np.ndarray:
    """The simulator's true probabilities for the test applicants (the ceiling; evaluation only)."""
    applicants = pd.read_parquet(
        config.DATA_PROCESSED / "applicants.parquet",
        columns=["applicant_id", "hidden_default_probability"],
    )
    known = load_split(applicants, "test", allow_test=True).set_index("applicant_id")
    return test["applicant_id"].map(known["hidden_default_probability"]).to_numpy()


def write_report(report: dict, path: Path = config.TEST_REPORT_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return path
