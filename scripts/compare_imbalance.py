"""Step 3.5: compare ways of handling the class imbalance (about 20% default).

Run from the project root:  python scripts/compare_imbalance.py

For logistic regression and XGBoost (the two models carried forward from step 3.4):
  none       - train on the data as it is
  weights    - make each defaulter count more (class_weight / scale_pos_weight)
  smote      - invent extra synthetic defaulters until the classes are balanced
               (SMOTE runs inside the pipeline, so only training folds are resampled)

The winner for each model is the best CV PR-AUC, using the tie rule: a strategy has to
beat "none" by more than one fold standard deviation to be worth its side effects.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from mkopoguard import config
from mkopoguard.features.columns import TARGET
from mkopoguard.models.candidates import XGBOOST, logistic, xgboost
from mkopoguard.models.experiment import load_training_data, results_table, run_experiment
from mkopoguard.tracking import setup_tracking

STEP = "3.5"
SEED = config.SIMULATION_SEED


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()
    ratio = float((train[TARGET] == 0).sum() / (train[TARGET] == 1).sum())
    print(f"Non-defaulters per defaulter in training: {ratio:.2f}")

    weighted_xgb = xgboost()
    weighted_xgb.set_params(scale_pos_weight=ratio)
    experiments = [
        ("logistic", "none", logistic(), {}, None),
        (
            "logistic",
            "weights",
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"class_weight": "balanced"},
            None,
        ),
        ("logistic", "smote", logistic(), {}, SMOTE(random_state=SEED)),
        ("xgboost", "none", xgboost(), XGBOOST, None),
        ("xgboost", "weights", weighted_xgb, {**XGBOOST, "scale_pos_weight": ratio}, None),
        ("xgboost", "smote", xgboost(), XGBOOST, SMOTE(random_state=SEED)),
    ]

    results = []
    for model_name, strategy, model, params, sampler in experiments:
        name = f"{model_name}_{strategy}"
        print(f"Training {name}...")
        result = run_experiment(
            name, model, train, validation, STEP, {**params, "imbalance": strategy}, sampler=sampler
        )
        results.append({**result, "model_family": model_name, "strategy": strategy})

    table = results_table(results)
    table.round(4).to_csv(config.RESULTS_DIR / "imbalance.csv")

    rows = []
    for _family, group in table.groupby("model_family"):
        none = group[group["strategy"] == "none"].iloc[0]
        for name, row in group.iterrows():
            gain = row["cv_pr_auc"] - none["cv_pr_auc"]
            rows.append(
                {
                    "run": name,
                    "cv_pr_auc": row["cv_pr_auc"],
                    "gain_vs_none": gain,
                    "worth_it": gain > none["cv_pr_auc_std"],
                    "val_pr_auc": row["val_pr_auc"],
                    "val_ece": row["val_ece"],
                    "val_approval_rate": row["val_approval_rate"],
                }
            )
    print(pd.DataFrame(rows).set_index("run").round(3).to_string())
    print(f"Saved {config.RESULTS_DIR / 'imbalance.csv'}")


if __name__ == "__main__":
    main()
