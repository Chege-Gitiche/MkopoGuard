"""Step 3.6: tune logistic regression and XGBoost with Optuna.

Rules that keep tuning honest:
- Every trial is scored by 5-fold cross-validation on the TRAINING set only. The validation
  set is used once at the end, as a check, and the test set is never touched.
- Trial 0 is always the untuned model, so "how much did tuning help?" is measured inside the
  same study, on the same folds.
- Tuned settings are only adopted if they beat the untuned ones by more than one fold
  standard deviation (evaluation plan, rule 6). Otherwise the simpler defaults stay.
- A seeded sampler run one trial at a time gives the same trials every run.
"""

import json
from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mkopoguard import config
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models.candidates import SEED, XGBOOST
from mkopoguard.models.experiment import cross_validate

# Starting points (trial 0). Logistic uses the saga solver so it can also try L1 (lasso)
# shrinkage: l1_ratio=0 is the plain L2 model from step 3.2, 1 is pure lasso.
DEFAULTS = {
    "logistic": {"C": 1.0, "l1_ratio": 0.0},
    # step 3.4's settings, plus XGBoost's own defaults for the two extra knobs we search
    "xgboost": {**XGBOOST, "reg_lambda": 1.0, "gamma": 0.0},
}


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    """The search space for each model. Ranges are wide but realistic for 13,000 rows."""
    if name == "logistic":
        return {
            # C = 1 / regularisation strength; small C = simpler model. Above ~3 saga
            # gets slow and the model is barely regularised anyway.
            "C": trial.suggest_float("C", 1e-3, 3.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # depth 1 = each tree uses one feature, i.e. an additive model like logistic
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    raise ValueError(f"No search space for {name!r}; expected 'logistic' or 'xgboost'")


def build_model(name: str, params: dict, seed: int = SEED):
    """A fresh, untrained model with the given settings."""
    if name == "logistic":
        # saga visits rows in a random order, so it gets a seed too
        return LogisticRegression(**params, solver="saga", max_iter=3000, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(
            **params, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed
        )
    raise ValueError(f"Unknown model {name!r}; expected 'logistic' or 'xgboost'")


def objective(name: str, train: pd.DataFrame):
    """Optuna objective: mean cross-validated PR-AUC on the training set (higher is better)."""
    columns = input_columns()

    def score(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name)
        cv = cross_validate(make_pipeline(build_model(name, params)), train, columns)
        trial.set_user_attr("cv_pr_auc_std", float(cv["pr_auc"].std()))
        trial.set_user_attr("cv_ece", float(cv["ece"].mean()))
        return float(cv["pr_auc"].mean())

    return score


def tune(
    name: str, train: pd.DataFrame, n_trials: int, seed: int = config.TUNING_SEED, callbacks=None
) -> optuna.Study:
    """Run one study. Trial 0 is the untuned default, so the study measures its own gain."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{name}-tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.enqueue_trial(DEFAULTS[name])
    study.optimize(objective(name, train), n_trials=n_trials, n_jobs=1, callbacks=callbacks)
    return study


def summarise(name: str, study: optuna.Study) -> dict:
    """Default vs best trial, and the tie-rule verdict."""
    default, best = study.trials[0], study.best_trial
    gain = best.value - default.value
    std = default.user_attrs["cv_pr_auc_std"]
    worth_it = gain > std
    return {
        "model": name,
        "n_trials": len(study.trials),
        "default_cv_pr_auc": default.value,
        "best_cv_pr_auc": best.value,
        "gain": gain,
        "default_cv_std": std,
        "worth_it": worth_it,
        "best_trial": best.number,
        "best_params": best.params,
        "chosen_params": best.params if worth_it else DEFAULTS[name],
    }


def trials_table(name: str, study: optuna.Study) -> pd.DataFrame:
    """Every trial as one row: its settings, CV score and fold std."""
    rows = []
    for t in study.trials:
        rows.append(
            {
                "model": name,
                "trial": t.number,
                "cv_pr_auc": t.value,
                "cv_pr_auc_std": t.user_attrs.get("cv_pr_auc_std"),
                "cv_ece": t.user_attrs.get("cv_ece"),
                **{f"param_{k}": v for k, v in t.params.items()},
            }
        )
    table = pd.DataFrame(rows)
    table["best_so_far"] = table["cv_pr_auc"].cummax()
    return table


def save_chosen(summaries: list[dict], path: Path = config.TUNED_PARAMS_FILE) -> Path:
    """Write the settings later steps should use (tuned only where tuning earned it)."""
    chosen = {
        s["model"]: {"params": s["chosen_params"], "tuned": bool(s["worth_it"])} for s in summaries
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def chosen_model(name: str, path: Path = config.TUNED_PARAMS_FILE, seed: int = SEED):
    """The model later steps (3.7 onwards) train: built from the settings step 3.6 chose."""
    chosen = json.loads(path.read_text(encoding="utf-8"))
    if name not in chosen:
        raise KeyError(f"{name!r} not in {path}; run scripts/tune_models.py first")
    return build_model(name, chosen[name]["params"], seed)
