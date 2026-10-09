"""Step 3.3: the baselines every later model must beat.

Run from the project root:  python scripts/train_baselines.py

1. dummy                  - predicts the training default rate for everyone (no skill)
2. logistic               - plain logistic regression
3. logistic_balanced      - logistic regression with class weights (defaulters count ~4x)
4. logistic_with_country  - plain logistic regression plus country_code

Results go to MLflow and docs/results/baselines.csv.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from mkopoguard import config
from mkopoguard.models.experiment import load_training_data, results_table, run_experiment
from mkopoguard.tracking import setup_tracking

STEP = "3.3"


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()
    print(f"Train {len(train):,} | validation {len(validation):,} | test stays locked")

    experiments = [
        ("dummy", DummyClassifier(strategy="prior"), {"strategy": "prior"}, False),
        ("logistic", LogisticRegression(max_iter=3000), {"C": 1.0}, False),
        (
            "logistic_balanced",
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"C": 1.0, "class_weight": "balanced"},
            False,
        ),
        ("logistic_with_country", LogisticRegression(max_iter=3000), {"C": 1.0}, True),
    ]
    results = []
    for name, model, params, with_country in experiments:
        print(f"Training {name}...")
        results.append(run_experiment(name, model, train, validation, STEP, params, with_country))

    table = results_table(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(config.RESULTS_DIR / "baselines.csv")
    columns = ["cv_pr_auc", "cv_pr_auc_std", "val_pr_auc", "val_roc_auc", "val_ece"]
    print(table[columns].round(3).to_string())
    print(f"Saved {config.RESULTS_DIR / 'baselines.csv'}")


if __name__ == "__main__":
    main()
