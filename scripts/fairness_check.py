"""Step 4.4: the fairness check of the champion across gender, age group and country.

Run from the project root:  python scripts/fairness_check.py
Takes about 30 seconds.

Uses out-of-fold predictions for the 15,910 development applicants: every applicant is scored
by a copy of the champion that never saw them (5-fold CV), so no group looks better because the
model memorised it. The test set stays closed. Decisions use the step 3.11 cost rule.

Outputs: docs/results/fairness_by_group.csv, docs/results/fairness_same_risk.csv,
         docs/results/fairness_summary.json, docs/img/fairness_01_approval.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard import fairness as fa
from mkopoguard.bundle import load_bundle
from mkopoguard.models import decision as dec
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.calibration import (
    calibrated_model,
    chosen_calibration,
    group_calibration,
    out_of_fold,
)
from mkopoguard.models.experiment import load_training_data
from mkopoguard.models.tuning import chosen_model
from mkopoguard.viz import BLUE, GREY, TEXT, _finish, apply_style, save

RESULTS = config.RESULTS_DIR
CHART = config.PROJECT_ROOT / "docs" / "img" / "fairness_01_approval.png"
TITLES = {"gender": "Gender", "age_group": "Age group", "country": "Country"}


def true_probability(frame: pd.DataFrame) -> np.ndarray:
    """The simulator's true probability of default (evaluation only, never a model input)."""
    applicants = pd.read_parquet(
        config.DATA_PROCESSED / "applicants.parquet",
        columns=["applicant_id", "hidden_default_probability"],
    ).set_index("applicant_id")
    return frame["applicant_id"].map(applicants["hidden_default_probability"]).to_numpy()


def plot(tables: dict[str, pd.DataFrame]) -> None:
    apply_style()
    heights = [len(tables[a]) + 1 for a in fa.ATTRIBUTES]
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 11), sharex=True, gridspec_kw={"height_ratios": heights}
    )
    for ax, attr in zip(axes, fa.ATTRIBUTES, strict=True):
        table = tables[attr]
        if attr == "country":
            table = table.sort_values("approval_rate")
        y = np.arange(len(table))
        colours = [GREY if r else BLUE for r in table["reference"]]
        ax.barh(y, table["approval_rate"], color=colours, height=0.65)
        ax.scatter(
            table["oracle_approval_rate"], y, color=TEXT, s=90, zorder=3, marker="|", linewidths=2
        )
        line = fa.FOUR_FIFTHS * table.loc[table["reference"], "approval_rate"].iloc[0]
        ax.axvline(line, color=TEXT, linestyle="--", linewidth=0.9)
        labels = [
            f"{g}  {'(below 4/5)' if flag else ''}"
            for g, flag in zip(table.index, table["adverse_impact"], strict=True)
        ]
        ax.set_yticks(y, labels, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_ylabel(TITLES[attr], fontsize=9)
    axes[-1].set_xlabel(
        "Approval rate. Bar = model, | = perfect-information oracle, "
        "dashed = 4/5 of the most-approved group (grey)",
        fontsize=8.5,
    )
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _finish(
        axes[0],
        "Women and over-50s are approved less, by more than their true risk explains",
        "Out-of-fold decisions for 15,910 development applicants",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)
    model = calibrated_model(chosen_model("logistic"), chosen_calibration())
    print("Scoring every development applicant out-of-fold (5 fits)...")
    p = out_of_fold(development, model, drop=chosen_drop())

    policy = dec.load_policy(config.DECISION_POLICY_FILE)
    terms = development["term_months"].astype(int)
    threshold = terms.map(policy["thresholds_by_term_months"]).to_numpy()
    true_p = true_probability(development)
    decline, oracle_decline = p >= threshold, true_p >= threshold
    y = development["defaulted"].to_numpy()
    groups = fa.audit_columns(development)

    # sanity: the out-of-fold decisions are the shipped model's decisions for almost everyone
    shipped = load_bundle().predict_proba(development) >= threshold
    agreement = float(np.mean(shipped == decline))

    tables, rows = {}, []
    for attr in fa.ATTRIBUTES:
        table = fa.audit(groups[attr], y, decline, oracle_decline)
        calibration = group_calibration(development.assign(**groups), p, attr)
        table["mean_predicted"] = calibration["expected"]
        table["mean_true_probability"] = (
            pd.Series(true_p).groupby(groups[attr].to_numpy(), observed=True).mean()
        )
        table["calibration_z"] = calibration["z"]
        table["calibration_flagged"] = calibration["flagged"]
        tables[attr] = table
        rows.append(table.assign(attribute=attr, subset="all"))
    for name, has in (("statement", 1), ("thin_file", 0)):  # is the gender gap only thin-file?
        mask = (development["has_statement"] == has).to_numpy()
        table = fa.audit(groups["gender"][mask], y[mask], decline[mask], oracle_decline[mask])
        rows.append(table.assign(attribute="gender", subset=name))
    by_group = pd.concat(rows).reset_index().set_index(["attribute", "subset", "group"])
    by_group.round(4).to_csv(RESULTS / "fairness_by_group.csv")

    same_risk = pd.concat(
        {
            "gender": fa.same_risk_gap(groups["gender"], decline, true_p, "male"),
            "age_group": fa.same_risk_gap(
                groups["age_group"].astype(str), decline, true_p, "25-34"
            ),
        },
        axis=1,
    )
    same_risk.round(4).to_csv(RESULTS / "fairness_same_risk.csv")
    plot(tables)

    flags = {
        attr: {
            flag: sorted(tables[attr].index[tables[attr][flag]].astype(str))
            for flag in ("adverse_impact", "fpr_flag", "model_adds_gap", "calibration_flagged")
        }
        for attr in fa.ATTRIBUTES
    }
    summary = {
        "predictions": "out-of-fold, 15,910 development applicants (test set not used)",
        "decision_agreement_with_shipped_model": round(agreement, 4),
        "rules": {
            "four_fifths": fa.FOUR_FIFTHS,
            "max_fpr_gap": fa.MAX_FPR_GAP,
            "interval": "bootstrap, Bonferroni-widened for the number of groups",
        },
        "flags": flags,
        "same_risk_average_decline_gap": {
            f"{attr}: {col}": round(float(v), 4)
            for (attr, col), v in same_risk.loc["average"].dropna().items()
        },
    }
    (RESULTS / "fairness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    cols = [
        "n",
        "default_rate",
        "approval_rate",
        "oracle_approval_rate",
        "approval_ratio",
        "fpr",
        "fpr_gap",
        "excess_approval_gap",
        "excess_fpr_gap",
        "mean_predicted",
        "mean_true_probability",
    ]
    print(f"Out-of-fold decisions agree with the shipped model for {agreement:.1%}")
    for attr in fa.ATTRIBUTES:
        print(f"\n{TITLES[attr]}:")
        print(tables[attr][cols].round(3).to_string())
        for flag, names in flags[attr].items():
            print(f"  {flag}: {names or 'none'}")
    print("\nGender, statement holders vs thin-file:")
    split = by_group.loc[("gender", ["statement", "thin_file"]), cols[:9]]
    print(split.round(3).to_string())
    print("\nAmong applicants with the SAME true risk, extra share declined:")
    print(same_risk.loc["average"].round(3).to_string())
    print("\nSaved fairness_by_group.csv, fairness_same_risk.csv, fairness_summary.json")


if __name__ == "__main__":
    main()
