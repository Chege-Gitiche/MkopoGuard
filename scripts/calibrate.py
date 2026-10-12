"""Step 3.10: check the champion's probabilities and decide whether to recalibrate them.

Run from the project root:  python scripts/calibrate.py
Takes about 3-5 minutes.

Three models, each with no calibration, sigmoid (Platt) and isotonic calibration, all on
out-of-fold CV predictions:
- logistic           the champion (settings from 3.6, feature groups from 3.7)
- logistic_weighted  POSITIVE CONTROL: class weights (step 3.5) make it badly calibrated on
                     purpose, so we can see that calibration repairs it when it is needed
- xgboost            the challenger, for comparison

Rule 8 decides the champion's method. The chosen version is then checked per country and
for thin-file vs statement applicants, and once on the validation set.

Outputs: docs/results/calibration.csv, docs/results/calibration_by_group.csv,
         docs/results/calibration.json, docs/img/models_05_calibration.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mkopoguard import config
from mkopoguard.models import calibration as cal
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.experiment import load_training_data, run_experiment
from mkopoguard.models.metrics import calibration_table
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import setup_tracking
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.10"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_05_calibration.png"
MIN_BIN = 50  # bins with fewer applicants are left off the chart (still in the CSV)
MODELS = {
    "logistic": lambda: chosen_model("logistic"),
    "logistic_weighted": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "xgboost": lambda: chosen_model("xgboost"),
}


def reliability(ax, y, p, label, colour, marker="o") -> None:
    table = calibration_table(y, p)
    table = table[table["applicants"] >= MIN_BIN]  # a handful of applicants is just noise
    sizes = 12 + 120 * table["applicants"] / table["applicants"].max()
    ax.plot(table["predicted"], table["actual"], color=colour, lw=1.5, label=label)
    ax.scatter(table["predicted"], table["actual"], s=sizes, color=colour, marker=marker, zorder=3)


def plot(train, oof: dict, groups: pd.DataFrame, method: str) -> None:
    y = train["defaulted"].to_numpy()
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8))

    left.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    reliability(left, y, oof[("logistic", method)], f"Champion ({method})", BLUE)
    reliability(left, y, oof[("logistic_weighted", "none")], "Class-weighted (no fix)", ORANGE)
    reliability(
        left, y, oof[("logistic_weighted", "sigmoid")], "Class-weighted + Platt", TEXT_MUTED, "s"
    )
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_xlabel("Predicted default rate (bin average)")
    left.set_ylabel("Actual default rate")
    left.legend(loc="upper left", fontsize=8.5)
    _finish(
        left,
        "The champion's probabilities can be taken at face value",
        "Out-of-fold CV predictions in bins of 50+ applicants (dot size: count); dotted: perfect",
        "none",
    )

    top = max(groups["expected"].max(), groups["observed"].max()) + 0.03
    right.plot([0, top], [0, top], color=GREY, lw=1, ls=":")
    colours = [BLUE if code == config.TARGET_COUNTRY else GREY for code in groups.index]
    right.scatter(groups["expected"], groups["observed"], color=colours, s=30, zorder=3)
    for code, row in groups.iterrows():
        right.annotate(
            code,
            (row["expected"], row["observed"]),
            fontsize=7.5,
            color=TEXT_MUTED,
            xytext=(3, 3),
            textcoords="offset points",
        )
    right.set_xlim(0.1, top)
    right.set_ylim(0.1, top)
    right.set_xlabel("Predicted default rate (average)")
    right.set_ylabel("Actual default rate")
    _finish(
        right,
        "No country is off by more than chance",
        "Each dot is one country; blue: Kenya",
        "none",
    )
    fig.tight_layout()
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()
    drop = chosen_drop()
    y = train["defaulted"].to_numpy()

    rows, oof, folds = [], {}, {}
    for name, factory in MODELS.items():
        for method in config.CALIBRATION_METHODS:
            print(f"Out-of-fold predictions: {name} / {method}...")
            p = cal.out_of_fold(train, cal.calibrated_model(factory(), method), drop)
            oof[(name, method)], folds[(name, method)] = p, cal.fold_metrics(train, p)
            rows.append({"model": name, "method": method, **cal.summary(y, p)})
    table = pd.DataFrame(rows).set_index(["model", "method"])

    for name in MODELS:
        for method in config.CALIBRATION_METHODS[1:]:
            gain = cal.calibration_gain(folds[(name, "none")], folds[(name, method)])
            for key, value in {**gain, "worth_it": cal.worth_calibrating(gain)}.items():
                table.loc[(name, method), key] = value
    table.round(4).to_csv(config.RESULTS_DIR / "calibration.csv")
    show = ["pr_auc", "brier", "ece", "slope", "intercept", "mean_predicted", "ece_gain"]
    print(table[show + ["worth_it"]].round(4).to_string())

    # The champion's method: the biggest ECE cut among methods that pass rule 8, else none
    passing = [
        m for m in config.CALIBRATION_METHODS[1:] if bool(table.loc[("logistic", m), "worth_it"])
    ]
    method = max(passing, key=lambda m: table.loc[("logistic", m), "ece_gain"], default="none")
    cal.save_choice(method)
    print(f"Champion calibration (rule 8): {method}")

    p = oof[("logistic", method)]
    statement = train.assign(
        statement=np.where(train["has_statement"] == 1, "statement", "thin_file")
    )
    by_country = cal.group_calibration(train, p, "country_code")
    by_statement = cal.group_calibration(statement, p, "statement")
    pd.concat({"country": by_country, "statement": by_statement}, names=["by"]).round(4).to_csv(
        config.RESULTS_DIR / "calibration_by_group.csv"
    )
    print(by_country.round(3).to_string())
    print(by_statement.round(3).to_string())
    print(
        f"Countries flagged (|z| > {cal.flag_z(len(by_country)):.2f}): "
        f"{list(by_country.index[by_country['flagged']]) or 'none'}"
    )
    plot(train, oof, by_country, method)

    result = run_experiment(
        f"logistic_calibration_{method}",
        cal.calibrated_model(chosen_model("logistic"), method),
        train,
        validation,
        STEP,
        {"calibration": method},
        drop=drop,
    )
    check = cal.summary(validation["defaulted"].to_numpy(), result["p_val"])
    print("Validation check: " + ", ".join(f"{k} {v:.3f}" for k, v in check.items()))
    print(f"Saved calibration.csv, calibration_by_group.csv, calibration.json and {CHART.name}")


if __name__ == "__main__":
    main()
