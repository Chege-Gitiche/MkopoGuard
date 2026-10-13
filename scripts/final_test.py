"""Step 3.12: open the locked test set ONCE and score the frozen champion.

Run from the project root, after committing your work:  python scripts/final_test.py
Takes about 1 minute.

Refuses to run with uncommitted changes, or a second time unless you give a reason:
    python scripts/final_test.py --rerun-reason "why the test set must be opened again"
Every opening is logged in docs/results/test_runs.log.

1. Fingerprint the frozen decisions (3.6 settings, 3.7 features, 3.10 calibration, 3.11 rule).
2. Refit the champion on training + validation applicants.
3. Score the 2,808 test applicants: ranking with 95% intervals, calibration, the decision
   rule's profit, and every subgroup; compare with what validation predicted.
4. Log the final model to MLflow (evaluated_on = test) for packaging in step 3.13.

Outputs: docs/results/test_report.json, docs/results/test_by_group.csv,
         docs/results/test_runs.log, docs/img/models_08_final_test.png
"""

import argparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.data.split import load_split
from mkopoguard.features.columns import TARGET
from mkopoguard.features.preprocess import input_columns
from mkopoguard.models import final
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.experiment import load_training_data
from mkopoguard.models.final import RANKING
from mkopoguard.models.metrics import FAST_METRICS, assert_no_leakage_alarm, calibration_table
from mkopoguard.tracking import git_commit, log_pipeline, setup_tracking, start_run
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.12"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_08_final_test.png"
LABELS = {"all": "All", "kenya": "Kenya", "statement": "Statement", "thin_file": "Thin-file"}
pr_auc = FAST_METRICS["pr_auc"]


def plot(groups: pd.DataFrame, validation: dict, y, p, ceiling: float) -> None:
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))

    x = np.arange(len(LABELS))
    test = groups.loc[list(LABELS)]
    err = [test["pr_auc"] - test["pr_auc_low"], test["pr_auc_high"] - test["pr_auc"]]
    left.errorbar(
        x,
        test["pr_auc"],
        yerr=err,
        fmt="o",
        color=BLUE,
        capsize=4,
        ms=7,
        label="Test set (95% interval)",
    )
    left.scatter(
        x + 0.15,
        [validation[g] for g in LABELS],
        color=ORANGE,
        marker="D",
        s=30,
        zorder=3,
        label="Validation set (what we expected)",
    )
    left.scatter(
        x,
        test["default_rate"],
        color=GREY,
        marker="_",
        s=300,
        label="No-skill baseline (default rate)",
    )
    left.axhline(ceiling, color=GREY, lw=1, ls=":")
    left.text(-0.3, ceiling + 0.01, "ceiling: true probabilities", color=TEXT_MUTED, fontsize=8.5)
    left.set_xticks(x, [f"{LABELS[g]}\n(n={int(test.loc[g, 'n'])})" for g in LABELS])
    left.set_ylim(0, 0.85)
    left.set_ylabel("PR-AUC")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper right", fontsize=8.5)
    _finish(
        left,
        "The test set confirms what validation predicted",
        "Champion PR-AUC on 2,808 never-seen applicants, by group",
        "none",
    )

    table = calibration_table(y, p)
    table = table[table["applicants"] >= 30]
    right.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    right.plot(table["predicted"], table["actual"], color=BLUE, lw=1.5)
    right.scatter(
        table["predicted"],
        table["actual"],
        color=BLUE,
        zorder=3,
        s=12 + 120 * table["applicants"] / table["applicants"].max(),
    )
    right.set_xlim(0, 0.8)
    right.set_ylim(0, 0.8)
    right.set_xlabel("Predicted default rate (bin average)")
    right.set_ylabel("Actual default rate")
    _finish(
        right,
        "Probabilities stay honest on new applicants",
        "Test set, bins of 30+ applicants (dot size: count); dotted: perfect",
        "none",
    )
    fig.tight_layout()
    save(fig, CHART)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen champion on the test set.")
    parser.add_argument("--rerun-reason", help="required if the test set was opened before")
    args = parser.parse_args()

    setup_tracking()
    commit = git_commit()
    try:
        final.check_can_open(commit, args.rerun_reason)
        decisions = final.frozen_decisions()
    except (PermissionError, FileNotFoundError) as error:
        raise SystemExit(f"Not opening the test set: {error}") from error
    print(f"Commit {commit}; frozen decisions: {decisions}")

    # Validation scores of the train-only champion: what we expect the test to show
    train, validation = load_training_data()
    columns = input_columns(drop=chosen_drop())
    expected_model = final.final_pipeline().fit(train[columns], train[TARGET])
    p_val = expected_model.predict_proba(validation[columns])[:, 1]
    expected = final.group_report(validation, p_val)["pr_auc"].to_dict()

    # The final model: same settings, refitted on training + validation
    development = pd.concat([train, validation], ignore_index=True)
    model = final.final_pipeline().fit(development[columns], development[TARGET])

    # Open the test set: the one and only time
    final.log_opening(commit, args.rerun_reason)
    table = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    test = load_split(table, "test", allow_test=True)
    p = model.predict_proba(test[columns])[:, 1]
    y = test[TARGET].to_numpy()

    report = final.build_report(test, p)
    assert_no_leakage_alarm({m: report["ranking"][m]["value"] for m in ("pr_auc", "roc_auc")})
    ceiling = pr_auc(y, final.true_probability_on_test(test))
    report.update(
        {
            "opened_at_commit": commit,
            "rerun_reason": args.rerun_reason,
            "frozen_decisions": decisions,
            "trained_on": {"applicants": len(development), "splits": ["train", "validation"]},
            "validation_pr_auc_by_group": expected,
            "ceiling_pr_auc": ceiling,
        }
    )
    groups = pd.DataFrame(report["groups"]).T
    groups.round(4).to_csv(config.RESULTS_DIR / "test_by_group.csv")
    final.write_report(report)
    plot(groups, expected, y, p, ceiling)

    with start_run("champion_final", evaluated_on="test", step=STEP, frozen=str(decisions)):
        mlflow.log_params(
            {
                "trained_on": "train+validation",
                "n_train": len(development),
                "n_features_in": len(columns),
            }
        )
        mlflow.log_metrics({f"test_{m}": report["ranking"][m]["value"] for m in RANKING})
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in report["calibration"].items() if k != "default_rate"}
        )
        rule = report["decision"]["cost_rule"]
        mlflow.log_metrics({f"test_{k}": v for k, v in rule.items() if k != "policy"})
        mlflow.log_text(groups.to_csv(), "test_by_group.csv")
        mlflow.log_dict(report, "test_report.json")
        log_pipeline(model)

    rk, cal = report["ranking"], report["calibration"]
    print(f"Test set: {report['n']} applicants, default rate {report['default_rate']:.3f}")
    for m in RANKING:
        print(f"  {m}: {rk[m]['value']:.3f} (95% interval {rk[m]['low']:.3f}-{rk[m]['high']:.3f})")
    print(f"  ceiling PR-AUC (true probabilities): {ceiling:.3f}")
    print(
        f"  calibration: ECE {cal['ece']:.3f}, slope {cal['slope']:.2f}, "
        f"intercept {cal['intercept']:.2f}, mean predicted {cal['mean_predicted']:.3f}"
    )
    d = report["decision"]
    gain = d["gain_vs_approve_everyone_per_1000"]
    print(
        f"  cost rule: approves {d['cost_rule']['approval_rate']:.1%}, defaults among approved "
        f"{d['cost_rule']['default_rate_approved']:.1%}, profit "
        f"KSh {d['cost_rule']['profit_per_1000_applicants_kes'] / 1e6:.2f}M per 1,000 "
        f"(+{gain['difference'] / 1e6:.2f}M vs approving everyone, "
        f"95% interval {gain['low'] / 1e6:+.2f}M to {gain['high'] / 1e6:+.2f}M)"
    )
    show = ["n", "default_rate", "pr_auc", "pr_auc_low", "pr_auc_high", "ece", "approval_rate"]
    print(groups[show].round(3).to_string())
    inside = {
        g: groups.loc[g, "pr_auc_low"] <= v <= groups.loc[g, "pr_auc_high"]
        for g, v in expected.items()
    }
    print(f"Validation PR-AUC inside the test interval: {inside}")
    print(
        f"Saved {config.TEST_REPORT_FILE.name}, test_by_group.csv and {CHART.name}; "
        f"logged champion_final to MLflow"
    )


if __name__ == "__main__":
    main()
