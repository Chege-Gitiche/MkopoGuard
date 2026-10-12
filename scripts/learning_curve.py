"""Step 3.8: learning curves for the champion (logistic) and the challenger (XGBoost).

Run from the project root:  python scripts/learning_curve.py
Takes about 1-3 minutes.

Both models use the settings chosen in 3.6 and the feature groups kept in 3.7. Each is
trained on 1/16, 1/8, 1/4, 1/2 and all of every CV fold's training rows, and scored on the
full held-out fold. The question: would more applicants improve the model, or has it
learned all the signal the inputs carry?

Outputs: docs/results/learning_curve.csv, docs/results/learning_curve_points.csv,
         docs/img/models_04_learning_curve.png, one MLflow run per model
"""

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.ticker import FuncFormatter

from mkopoguard import config
from mkopoguard.models import learning_curve as lc
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.experiment import load_training_data
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import setup_tracking, start_run
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.8"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_04_learning_curve.png"
MODELS = {"logistic": ("Logistic (champion)", BLUE), "xgboost": ("XGBoost", ORANGE)}


def plot(summaries: dict[str, pd.DataFrame], top: float) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, table in summaries.items():
        label, colour = MODELS[name]
        n, cv, std = table["n_train"], table["cv_pr_auc"], table["cv_pr_auc_std"].fillna(0)
        ax.plot(n, cv, color=colour, lw=2, marker="o", ms=4, label=f"{label}: unseen applicants")
        ax.fill_between(n, cv - std, cv + std, color=colour, alpha=0.12, lw=0)
        ax.plot(
            n,
            table["train_pr_auc"],
            color=colour,
            lw=1.2,
            ls="--",
            label=f"{label}: its own training applicants",
        )
    ax.axhline(top, color=GREY, lw=1.5, ls=":")
    ax.text(
        summaries["logistic"]["n_train"].iloc[0],
        top + 0.01,
        "Ceiling: the simulator's true probabilities",
        color=TEXT_MUTED,
        fontsize=9,
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(summaries["logistic"]["n_train"].to_list())
    ax.set_xlabel("Applicants the model was trained on (log scale: each step doubles)")
    ax.set_ylabel("PR-AUC")
    ax.legend(loc="upper right", fontsize=8.5)
    _finish(
        ax,
        "Logistic regression has nearly run out of things to learn",
        "PR-AUC by training size, averaged over 5 CV folds (band: run-to-run spread)",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, _ = load_training_data()  # the validation set isn't needed here
    drop = chosen_drop()

    top = lc.ceiling(train, lc.true_probability(train))
    print(
        f"Ceiling (true probabilities): PR-AUC {top['all']:.4f} (thin-file {top['thin_file']:.4f})"
    )

    summaries, points, verdicts = {}, [], []
    for name in MODELS:
        print(f"Learning curve for {name}...")
        pts = lc.curve_points(train, lambda name=name: chosen_model(name), drop=drop)
        table = lc.summarise(pts)
        gain = lc.projected_gain(table)
        helps = lc.more_data_helps(gain)
        summaries[name] = table
        points.append(pts.assign(model=name))
        verdicts.append(
            {
                "model": name,
                **gain,
                "more_data_helps": helps,
                "gap_to_ceiling": top["all"] - table["cv_pr_auc"].iloc[-1],
            }
        )

        with start_run(f"learning_curve_{name}", evaluated_on="cv", step=STEP):
            mlflow.log_params(
                {
                    "model": name,
                    "dropped_columns": len(drop),
                    "fractions": ",".join(map(str, config.LEARNING_CURVE_FRACTIONS)),
                }
            )
            for row in table.itertuples():
                mlflow.log_metric("cv_pr_auc", row.cv_pr_auc, step=row.n_train)
                mlflow.log_metric("train_pr_auc", row.train_pr_auc, step=row.n_train)
            mlflow.log_metrics({**gain, "ceiling_pr_auc": top["all"]})
            mlflow.log_text(table.to_csv(), "learning_curve.csv")

    summary = pd.concat(summaries, names=["model"])
    summary.round(4).to_csv(config.RESULTS_DIR / "learning_curve.csv")
    pd.concat(points).round(4).to_csv(config.RESULTS_DIR / "learning_curve_points.csv", index=False)
    plot(summaries, top["all"])

    print(summary.round(4).to_string())
    verdict = pd.DataFrame(verdicts).set_index("model")
    print(verdict.round(4).to_string())
    for name, row in verdict.iterrows():
        if row["more_data_helps"]:
            print(
                f"{name}: still learning - another doubling is projected to add "
                f"{row['next_gain']:.3f} PR-AUC"
            )
        else:
            print(
                f"{name}: plateau - another doubling is projected to add only "
                f"{row['next_gain']:.3f}; the remaining gap to the ceiling "
                f"({row['gap_to_ceiling']:.3f}) is information the inputs don't carry"
            )
    print(f"Saved learning_curve.csv, learning_curve_points.csv and {CHART.name}")


if __name__ == "__main__":
    main()
