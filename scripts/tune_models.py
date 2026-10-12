"""Step 3.6: tune logistic regression and XGBoost with Optuna.

Run from the project root:  python scripts/tune_models.py
Takes roughly 5-15 minutes depending on your CPU.

For each model: an Optuna study scored by 5-fold CV on the training set (trial 0 = the
untuned model), then the tie rule decides whether the tuned settings are worth adopting.
The chosen settings are then trained and checked on the validation set through the usual
harness (so they are logged to MLflow like every other run).

Outputs: docs/results/tuning.csv, docs/results/tuning_trials.csv,
         docs/results/tuned_params.json, docs/img/models_02_tuning.png
"""

import time

import matplotlib.pyplot as plt
import pandas as pd

from mkopoguard import config
from mkopoguard.models import tuning
from mkopoguard.models.experiment import load_training_data, run_experiment
from mkopoguard.tracking import setup_tracking
from mkopoguard.viz import BLUE, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.6"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_02_tuning.png"
LABELS = {"logistic": "Logistic", "xgboost": "XGBoost"}
COLOURS = {"logistic": BLUE, "xgboost": ORANGE}


def progress(name: str, n_trials: int, start: float):
    """Print a line every 10 trials so a long study doesn't look frozen."""

    def report(study, trial) -> None:
        done = trial.number + 1
        if done % 10 == 0 or done == n_trials:
            print(
                f"  {name}: {done}/{n_trials} trials, best CV PR-AUC {study.best_value:.4f} "
                f"({time.time() - start:.0f}s)"
            )

    return report


def plot_tuning(trials: pd.DataFrame) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, group in trials.groupby("model"):
        ax.plot(group["trial"], group["best_so_far"], color=COLOURS[name], lw=2, label=LABELS[name])
        ax.scatter(group["trial"], group["cv_pr_auc"], color=COLOURS[name], s=12, alpha=0.35)
        ax.axhline(group["cv_pr_auc"].iloc[0], color=COLOURS[name], lw=1, ls=":")
    # A few hopeless trials score far lower; cut them off so the top of the chart is readable
    floor = trials.groupby("model")["cv_pr_auc"].first().min() - 0.02
    ax.set_ylim(bottom=floor)
    ax.set_xlabel("Trial (0 = untuned settings, dotted line)")
    ax.set_ylabel("Cross-validated PR-AUC")
    ax.legend(loc="lower right")
    ax.text(
        1.0,
        -0.2,
        "Dots: each trial (very poor trials fall below the chart). Line: best so far.",
        transform=ax.transAxes,
        ha="right",
        color=TEXT_MUTED,
        fontsize=9,
    )
    _finish(
        ax,
        "Tuning lifts XGBoost to logistic's level, but no higher",
        "Best cross-validated PR-AUC found so far, per Optuna trial",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()

    summaries, trials = [], []
    for name, n_trials in config.TUNING_TRIALS.items():
        print(f"Tuning {name} ({n_trials} trials)...")
        study = tuning.tune(
            name, train, n_trials, callbacks=[progress(name, n_trials, time.time())]
        )
        summaries.append(tuning.summarise(name, study))
        trials.append(tuning.trials_table(name, study))

    trials = pd.concat(trials, ignore_index=True)
    trials.round(4).to_csv(config.RESULTS_DIR / "tuning_trials.csv", index=False)
    tuning.save_chosen(summaries)
    plot_tuning(trials)

    # Check on the validation set: untuned vs best-found settings, through the usual harness
    rows = []
    for s in summaries:
        name = s["model"]
        for label, params in (("default", tuning.DEFAULTS[name]), ("tuned", s["best_params"])):
            print(f"Validating {name}_{label}...")
            result = run_experiment(
                f"{name}_{label}",
                tuning.build_model(name, params),
                train,
                validation,
                STEP,
                {**params, "tuning": label, "n_trials": s["n_trials"]},
            )
            s[f"val_pr_auc_{label}"] = result["val_pr_auc"]
            s[f"val_ece_{label}"] = result["val_ece"]
            s[f"overfit_gap_{label}"] = result["overfit_gap"]
        rows.append({k: v for k, v in s.items() if k not in ("best_params", "chosen_params")})

    table = pd.DataFrame(rows).set_index("model")
    table.round(4).to_csv(config.RESULTS_DIR / "tuning.csv")

    show = ["default_cv_pr_auc", "best_cv_pr_auc", "gain", "default_cv_std", "worth_it"]
    print(table[show].round(4).to_string())
    check = ["val_pr_auc_default", "val_pr_auc_tuned", "overfit_gap_default", "overfit_gap_tuned"]
    print(table[check].round(4).to_string())
    for s in summaries:
        verdict = "ADOPT tuned settings" if s["worth_it"] else "keep defaults (gain is a tie)"
        print(f"{s['model']}: {verdict}. Best settings found: {s['best_params']}")
    print(
        f"Saved {config.RESULTS_DIR / 'tuning.csv'}, tuning_trials.csv, "
        f"{config.TUNED_PARAMS_FILE.name} and {CHART}"
    )


if __name__ == "__main__":
    main()
