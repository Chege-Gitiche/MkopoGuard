"""Step 3.4: train the tree-based candidates and compare them with the baseline.

Run from the project root:  python scripts/train_candidates.py
(Run scripts/train_baselines.py first: the comparison reads docs/results/baselines.csv.)

Each model goes through the same harness as the baselines (5-fold CV on train, then the
validation set). Results go to MLflow, docs/results/candidates.csv and an overfitting chart.
"""

import pandas as pd

from mkopoguard import config
from mkopoguard.models.candidates import CANDIDATES
from mkopoguard.models.experiment import load_training_data, results_table, run_experiment
from mkopoguard.tracking import setup_tracking
from mkopoguard.viz import apply_style, paired_bars, save

STEP = "3.4"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_01_overfitting.png"
LABELS = {
    "logistic": "Logistic",
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}


def plot_overfitting(baselines: pd.DataFrame, candidates: pd.DataFrame) -> None:
    both = pd.concat([baselines, candidates])
    table = both.loc[list(LABELS), ["train_pr_auc", "val_pr_auc"]]
    table = table.rename(columns={"train_pr_auc": "Training set", "val_pr_auc": "Validation set"})
    table.index = list(LABELS.values())
    apply_style()
    fig = paired_bars(
        table,
        "Tree models memorise the training data; logistic regression does not",
        "PR-AUC on applicants each model trained on vs applicants it has never seen",
        percent=False,
    )
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()
    baselines = pd.read_csv(config.RESULTS_DIR / "baselines.csv", index_col="name")
    baseline = baselines.loc["logistic"]

    results = []
    for name, factory, params in CANDIDATES:
        print(f"Training {name}...")
        results.append(run_experiment(name, factory(), train, validation, STEP, params))

    table = results_table(results)
    table.round(4).to_csv(config.RESULTS_DIR / "candidates.csv")
    plot_overfitting(baselines, table)

    # Tie rule (evaluation plan, rule 6): a gain smaller than one fold std is not a gain
    table["gain_vs_logistic"] = table["cv_pr_auc"] - baseline["cv_pr_auc"]
    table["beats_logistic"] = table["gain_vs_logistic"] > table["cv_pr_auc_std"]
    columns = ["cv_pr_auc", "gain_vs_logistic", "beats_logistic", "train_pr_auc", "val_pr_auc"]
    print(f"Baseline to beat: logistic, CV PR-AUC {baseline['cv_pr_auc']:.3f}")
    print(table[columns + ["overfit_gap"]].round(3).to_string())
    print(f"Saved {config.RESULTS_DIR / 'candidates.csv'} and {CHART}")


if __name__ == "__main__":
    main()
