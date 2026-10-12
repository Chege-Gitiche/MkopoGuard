"""Step 3.9: leave-one-country-out. Does the model work in a country it has never seen?

Run from the project root:  python scripts/country_transfer.py
Takes about 3-8 minutes (each model is trained 105 times).

For the champion (logistic) and the challenger (XGBoost), each training applicant is scored
by a model that has seen their country and by one trained without it (see
models/country_transfer.py). Rule 9 decides whether the model transfers to new countries.

Outputs: docs/results/country_transfer.csv, docs/results/country_transfer_calibration.csv,
         docs/img/models_06_country_transfer.png, one MLflow run per model
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.models import country_transfer as ct
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.experiment import load_training_data
from mkopoguard.models.metrics import FAST_METRICS
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import setup_tracking, start_run
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.9"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_06_country_transfer.png"
MODELS = ("logistic", "xgboost")
pr_auc = FAST_METRICS["pr_auc"]


def plot(table: pd.DataFrame) -> None:
    """Dumbbell chart for the champion: each country seen vs never seen."""
    order = table.sort_values("pr_auc_seen").index
    y = np.arange(len(order))
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, code in enumerate(order):
        seen, unseen = table.loc[code, ["pr_auc_seen", "pr_auc_unseen"]]
        ax.plot([seen, unseen], [i, i], color=GREY, lw=2, zorder=1)
    ax.scatter(
        table.loc[order, "pr_auc_seen"],
        y,
        color=GREY,
        s=28,
        zorder=2,
        label="Model has seen this country",
    )
    colours = [BLUE if c == config.TARGET_COUNTRY else ORANGE for c in order]
    ax.scatter(
        table.loc[order, "pr_auc_unseen"],
        y,
        color=colours,
        s=36,
        zorder=3,
        label="Model has NEVER seen this country",
    )
    ax.set_yticks(y, [f"{c}  (n={table.loc[c, 'n']})" for c in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC on that country's applicants")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(
        1.0,
        -0.1,
        "Blue dot: Kenya",
        transform=ax.transAxes,
        ha="right",
        color=TEXT_MUTED,
        fontsize=9,
    )
    _finish(
        ax,
        "The champion works as well in a country it has never seen",
        "Logistic regression, out-of-fold CV on the training set, per country",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, _ = load_training_data()  # per-country validation sets (~140) are too small
    drop = chosen_drop()

    tables, calibrations, verdicts = {}, {}, []
    for name in MODELS:
        print(f"Leave-one-country-out for {name} (5 folds x 20 countries)...")
        preds = ct.seen_and_unseen(train, lambda name=name: chosen_model(name), drop)
        table, calib = ct.country_table(preds), ct.unseen_calibration(preds)
        tables[name], calibrations[name] = table, calib
        y = preds["defaulted"]
        summary = {
            "model": name,
            "pooled_pr_auc_seen": pr_auc(y, preds["p_seen"]),
            "pooled_pr_auc_unseen": pr_auc(y, preds["p_unseen"]),
            "median_loss": table["loss"].median(),
            "worst_loss": table["loss"].max(),
            "worst_country": table["loss"].idxmax(),
            "kenya_loss": table.loc[config.TARGET_COUNTRY, "loss"],
            "countries_losing_clearly": int(table["clear_loss"].sum()),
            "countries_miscalibrated": int(calib["flagged"].sum()),
            "transfers": ct.transfers(table, calib),
        }
        verdicts.append(summary)
        with start_run(f"country_transfer_{name}", evaluated_on="cv", step=STEP):
            mlflow.log_params({"model": name, "dropped_columns": len(drop)})
            mlflow.log_metrics(
                {k: float(v) for k, v in summary.items() if k not in ("model", "worst_country")}
            )
            mlflow.log_text(table.to_csv(), "country_transfer.csv")
            mlflow.log_text(calib.to_csv(), "country_transfer_calibration.csv")

    pd.concat(tables, names=["model"]).round(4).to_csv(config.RESULTS_DIR / "country_transfer.csv")
    pd.concat(calibrations, names=["model"]).round(4).to_csv(
        config.RESULTS_DIR / "country_transfer_calibration.csv"
    )
    plot(tables["logistic"])

    for name in MODELS:
        show = ["n", "default_rate", "pr_auc_seen", "pr_auc_unseen", "loss", "loss_low"]
        print(f"\n{name}:")
        print(tables[name][show + ["loss_high", "clear_loss"]].round(3).to_string())
        calib = calibrations[name]
        print(
            "Never-seen calibration, largest |z|: "
            f"{calib['z'].abs().idxmax()} (z = {calib['z'].abs().max():.2f}); "
            f"flagged: {list(calib.index[calib['flagged']]) or 'none'}"
        )
    verdict = pd.DataFrame(verdicts).set_index("model")
    print("\n" + verdict.round(4).T.to_string())
    for name, row in verdict.iterrows():
        outcome = "TRANSFERS to new countries" if row["transfers"] else "does NOT transfer"
        print(f"{name}: {outcome} (rule 9)")
    print(f"Saved country_transfer.csv, country_transfer_calibration.csv and {CHART.name}")


if __name__ == "__main__":
    main()
