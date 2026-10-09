"""Step 3.4: the candidate models, with sensible starting settings (tuned in step 3.6).

Each entry is (name, factory, params). factory(seed) builds a fresh, untrained model, so
every experiment and every test starts from the same place. Settings lean conservative
(shallow trees, a minimum number of applicants per leaf) because 13,000 rows is small for
boosting and over-confident trees give poor probabilities.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mkopoguard import config

SEED = config.SIMULATION_SEED

RANDOM_FOREST = {
    "n_estimators": 300,
    "min_samples_leaf": 20,  # each leaf averages at least 20 applicants: smoother probabilities
    "max_features": "sqrt",
}
XGBOOST = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
LIGHTGBM = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def logistic(seed: int = SEED):
    return LogisticRegression(max_iter=3000)


def random_forest(seed: int = SEED):
    # n_jobs=1: parallel prediction adds tree votes in a varying order, changing the last
    # decimal place between calls; one thread keeps predictions exactly reproducible
    return RandomForestClassifier(**RANDOM_FOREST, n_jobs=1, random_state=seed)


def xgboost(seed: int = SEED):
    return XGBClassifier(
        **XGBOOST, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed
    )


def lightgbm(seed: int = SEED):
    # deterministic + force_col_wise make LightGBM give identical results run after run
    return LGBMClassifier(
        **LIGHTGBM, deterministic=True, force_col_wise=True, verbose=-1, random_state=seed
    )


CANDIDATES = [
    ("random_forest", random_forest, RANDOM_FOREST),
    ("xgboost", xgboost, XGBOOST),
    ("lightgbm", lightgbm, LIGHTGBM),
]
