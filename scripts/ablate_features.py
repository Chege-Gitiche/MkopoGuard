"""Step 3.7: feature ablation. Which groups of inputs does the champion model need?

Run from the project root:  python scripts/ablate_features.py
Takes about 2-4 minutes (17 cross-validations plus two validation runs).

1. Remove each feature group in turn (and try each group on its own), on 5-fold CV.
2. Rule 7 decides which groups are needed (see models/ablation.py).
3. Drop all the unneeded groups together and check the combined drop on CV.
4. Check it on the validation set, per subgroup, with a paired bootstrap. If any subgroup
   loses clearly, nothing is dropped.

Outputs: docs/results/ablation.csv, docs/results/ablation_validation.csv,
         docs/results/feature_selection.json, docs/img/models_03_ablation.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.features.groups import FEATURE_GROUPS, SURVEY_GROUPS, columns_in
from mkopoguard.features.preprocess import input_columns
from mkopoguard.models import ablation
from mkopoguard.models.experiment import load_training_data, run_experiment, subgroups
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import setup_tracking
from mkopoguard.viz import BLUE, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.7"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_03_ablation.png"
LABELS = {
    "loan": "Loan terms",
    "profile": "Profile",
    "financial_access": "Financial access",
    "income_channels": "Income channels",
    "savings_resilience": "Savings & resilience",
    "borrowing_history": "Borrowing history",
    "transfers": "Transfers",
    "statement": "M-Pesa statement",
}


def champion():
    return chosen_model("logistic")


def plot_ablation(table: pd.DataFrame) -> None:
    order = table.sort_values("loss_all").index
    y = np.arange(len(order))
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y + 0.2, table.loc[order, "loss_all"], height=0.38, color=BLUE, label="All applicants")
    ax.barh(
        y - 0.2,
        table.loc[order, "loss_thin_file"],
        height=0.38,
        color=ORANGE,
        label="Thin-file applicants (no statement)",
    )
    ax.axvline(0, color=TEXT_MUTED, lw=0.8)
    ax.axvline(config.ABLATION_MIN_LOSS, color=TEXT_MUTED, lw=0.8, ls=":")
    names = [LABELS[g] + ("" if table.loc[g, "needed"] else "  (dropped)") for g in order]
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Loss in cross-validated PR-AUC when the group is removed")
    ax.legend(loc="lower right")
    _finish(
        ax,
        "Savings answers and the statement carry the model",
        "How much worse the model gets without each group (dotted line: smallest loss that counts)",
        "none",
    )
    save(fig, CHART)


def validation_check(train, validation, dropped: list[str]) -> pd.DataFrame:
    """Full vs reduced model on the validation set, per subgroup, with paired bootstrap CIs."""
    drop = columns_in(dropped)
    full = run_experiment("logistic_all_features", champion(), train, validation, STEP)
    reduced = run_experiment(
        "logistic_reduced",
        champion(),
        train,
        validation,
        STEP,
        {"dropped_groups": ",".join(dropped) or "none"},
        drop=drop,
    )
    y = validation["defaulted"].to_numpy()
    rows = {}
    for name, mask in subgroups(validation).items():
        rows[name] = ablation.paired_bootstrap(y[mask], full["p_val"][mask], reduced["p_val"][mask])
    table = pd.DataFrame(rows).T
    table["clear_loss"] = table["low"] > 0  # the whole interval says "worse"
    return table


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()

    print("Cross-validating the full model...")
    full = ablation.fold_scores(train, champion())
    all_score, thin_score = full["all"].mean(), full["thin_file"].mean()
    print(f"  full model: CV PR-AUC {all_score:.4f} (thin-file {thin_score:.4f})")

    print("Removing each group in turn (and trying each group alone)...")
    table = ablation.drop_one_table(train, champion, full)
    table.round(4).to_csv(config.RESULTS_DIR / "ablation.csv")
    plot_ablation(table)
    show = ["cv_pr_auc_without", "cv_pr_auc_only", "loss_all", "se_all"]
    show += ["loss_thin_file", "se_thin_file", "needed"]
    print(table[show].round(4).to_string())

    dropped = [g for g in FEATURE_GROUPS if not table.loc[g, "needed"]]
    print(f"Not needed (rule 7): {dropped or 'none'}")

    if dropped:
        combined = ablation.fold_scores(train, champion(), drop_groups=dropped)
        loss = ablation.paired_loss(full, combined)
        print(
            f"Combined drop on CV: PR-AUC {combined['all'].mean():.4f} "
            f"(thin-file {combined['thin_file'].mean():.4f}); "
            f"loss {loss['loss_all']:+.4f} all, {loss['loss_thin_file']:+.4f} thin-file"
        )
        if ablation.is_needed(loss):
            print("The groups matter TOGETHER, so nothing is dropped.")
            dropped = []

    if dropped:
        check = validation_check(train, validation, dropped)
        check.round(4).to_csv(config.RESULTS_DIR / "ablation_validation.csv")
        print("Validation check (loss = full - reduced, 95% paired bootstrap interval):")
        print(check.round(4).to_string())
        if check["clear_loss"].any():
            print("A subgroup loses clearly on validation, so nothing is dropped.")
            dropped = []

    ablation.save_selection(dropped)
    kept_columns = input_columns(drop=columns_in(dropped))
    survey_before = len(columns_in(SURVEY_GROUPS))
    survey_after = len(columns_in([g for g in SURVEY_GROUPS if g not in dropped]))
    print(f"Decision: drop {dropped or 'nothing'}")
    print(f"Model inputs: {len(input_columns())} -> {len(kept_columns)}")
    print(f"Survey questions on the application form: {survey_before} -> {survey_after}")
    print(f"Saved {config.FEATURE_SELECTION_FILE.name}, ablation.csv and {CHART.name}")


if __name__ == "__main__":
    main()
