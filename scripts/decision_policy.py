"""Step 3.11: set the approve / decline rule from the lender's costs, and check it.

Run from the project root:  python scripts/decision_policy.py
Takes about 1 minute.

1. Out-of-fold CV predictions for the champion (every training applicant scored by a model
   that never saw them).
2. Compare four ways of deciding on realised profit: approve everyone, the provisional 0.20
   threshold, the best single threshold found by trial and error, and the cost-based rule.
3. Check the cost-based rule is (near) optimal by scaling its thresholds up and down.
4. Sensitivity: how thresholds, approvals and profit move if the cost assumptions are off.
5. Check once on the validation set, with a bootstrap interval for the profit difference.

Outputs: docs/results/decision_policies.csv, docs/results/decision_sensitivity.csv,
         docs/results/decision_policy.json, docs/img/models_07_decision.png, an MLflow run
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.features.preprocess import input_columns, make_pipeline
from mkopoguard.models import decision as dec
from mkopoguard.models.ablation import chosen_drop
from mkopoguard.models.calibration import calibrated_model, chosen_calibration, out_of_fold
from mkopoguard.models.experiment import load_training_data, subgroups
from mkopoguard.models.tuning import chosen_model
from mkopoguard.tracking import setup_tracking, start_run
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

STEP = "3.11"
CHART = config.PROJECT_ROOT / "docs" / "img" / "models_07_decision.png"


def champion():
    return calibrated_model(chosen_model("logistic"), chosen_calibration())


def compare(frame: pd.DataFrame, p: np.ndarray, single: float) -> pd.DataFrame:
    y, amount, term = frame["defaulted"], frame["loan_amount_kes"], frame["term_months"]
    policies = {
        "approve_everyone": np.ones(len(p), dtype=bool),
        "provisional_0.20": p < config.PROVISIONAL_THRESHOLD,
        f"best_single_{single:.2f}": p < single,
        "cost_rule": dec.cost_rule(p, term),
    }
    rows = [dec.policy_summary(name, y, a, amount, term) for name, a in policies.items()]
    return pd.DataFrame(rows).set_index("policy")


def by_group(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Approval and defaults among approved, per subgroup and per loan term, under the rule."""
    approve = dec.cost_rule(p, frame["term_months"])
    y = frame["defaulted"].to_numpy()
    groups = {**subgroups(frame)}
    for term in config.TERM_OPTIONS:
        groups[f"term_{term}m"] = (frame["term_months"] == term).to_numpy()
    rows = {
        name: {
            "n": int(mask.sum()),
            "approval_rate": approve[mask].mean(),
            "default_rate_all": y[mask].mean(),
            "default_rate_approved": y[mask & approve].mean() if (mask & approve).any() else np.nan,
        }
        for name, mask in groups.items()
    }
    return pd.DataFrame(rows).T


def plot(frame: pd.DataFrame, p: np.ndarray, scale: pd.DataFrame, single: float) -> None:
    y, amount, term = frame["defaulted"], frame["loan_amount_kes"], frame["term_months"]
    grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
    per_1000 = [dec.realised_profit(y, p < t, amount, term).mean() * 1_000 / 1e6 for t in grid]
    rule = dec.realised_profit(y, dec.cost_rule(p, term), amount, term).mean() * 1_000 / 1e6
    everyone = dec.realised_profit(y, np.ones(len(p), bool), amount, term).mean() * 1_000 / 1e6

    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    left.plot(grid, per_1000, color=GREY, lw=2, label="One threshold for every loan")
    left.axhline(rule, color=BLUE, lw=2, label="Cost-based rule (threshold per loan term)")
    left.axhline(everyone, color=ORANGE, lw=1.5, ls="--", label="Approve everyone")
    left.axvline(config.PROVISIONAL_THRESHOLD, color=TEXT_MUTED, lw=0.8, ls=":")
    left.text(
        config.PROVISIONAL_THRESHOLD + 0.005,
        everyone,
        "provisional 0.20",
        color=TEXT_MUTED,
        fontsize=8.5,
        va="bottom",
    )
    left.set_xlabel("Decline when the probability of default is at least...")
    left.set_ylabel("Profit per 1,000 applicants (KES million)")
    left.legend(loc="lower right", fontsize=8.5)
    _finish(
        left,
        "Pricing risk by loan term beats any single cut-off",
        f"Realised profit on out-of-fold CV predictions (best single: {single:.2f})",
        "none",
    )

    right.plot(
        scale.index,
        scale["profit_kes"] / scale["profit_kes"].max(),
        color=BLUE,
        lw=2,
        marker="o",
        ms=3,
    )
    right.axvline(1.0, color=TEXT_MUTED, lw=0.8, ls=":")
    right.set_xlabel("Every cost-based threshold multiplied by...")
    right.set_ylabel("Profit, relative to the best")
    _finish(
        right,
        "The thresholds the costs predict are the ones that pay",
        "Scaling all thresholds up or down; 1.0 = exactly what the costs say",
        "none",
    )
    fig.tight_layout()
    save(fig, CHART)


def main() -> None:
    setup_tracking()
    train, validation = load_training_data()
    drop = chosen_drop()
    y, amount, term = train["defaulted"], train["loan_amount_kes"], train["term_months"]

    fee, funding = config.MONTHLY_FEE_RATE, config.MONTHLY_FUNDING_COST
    print(
        f"Cost assumptions: fee {fee:.1%}/month - funding {funding:.1%}/month = "
        f"net margin {dec.NET_MARGIN:.1%}/month; loss given default {config.LOSS_GIVEN_DEFAULT:.0%}"
    )
    print(
        "Thresholds by term: "
        + ", ".join(
            f"{t} month(s): decline at p >= {v:.3f}" for t, v in dec.thresholds_by_term().items()
        )
    )

    print("Out-of-fold predictions for the champion...")
    p = out_of_fold(train, champion(), drop)

    single = dec.best_single_threshold(y, p, amount, term)
    policies = compare(train, p, single)
    scale = dec.scale_check(y, p, amount, term)
    best_scale = float(scale["profit_kes"].idxmax())
    expected = dec.expected_profit(p, dec.cost_rule(p, term), amount, term).sum()
    realised = policies.loc["cost_rule", "profit_kes"]
    sens = dec.sensitivity(y, p, amount, term)

    policies.round(4).to_csv(config.RESULTS_DIR / "decision_policies.csv")
    sens.round(4).to_csv(config.RESULTS_DIR / "decision_sensitivity.csv", index=False)
    plot(train, p, scale, single)
    print(policies.round(3).to_string())
    print(f"Scale check: best profit at {best_scale:.2f} x the cost-based thresholds")
    print(
        f"Model's expected profit {expected / 1e6:.2f}M vs realised {realised / 1e6:.2f}M KES "
        f"(ratio {realised / expected:.2f})"
    )
    print(by_group(train, p).round(3).to_string())
    print(sens.round(3).to_string(index=False))

    # Validation check: train on all of train, decide on validation applicants
    columns = input_columns(drop=drop)
    pipeline = make_pipeline(champion(), drop=drop).fit(train[columns], train["defaulted"])
    p_val = pipeline.predict_proba(validation[columns])[:, 1]
    val = compare(validation, p_val, single)
    v_y, v_amount, v_term = (validation[c] for c in ("defaulted", "loan_amount_kes", "term_months"))
    rule_val = dec.cost_rule(p_val, v_term)
    vs_provisional = dec.profit_difference(
        v_y, rule_val, p_val < config.PROVISIONAL_THRESHOLD, v_amount, v_term
    )
    vs_everyone = dec.profit_difference(v_y, rule_val, np.ones(len(p_val), bool), v_amount, v_term)
    print("Validation check:")
    print(val.round(3).to_string())
    for label, d in (("provisional 0.20", vs_provisional), ("approving everyone", vs_everyone)):
        print(
            f"  cost rule vs {label}: {d['difference'] / 1e6:+.2f}M KES per 1,000 applicants "
            f"(95% interval {d['low'] / 1e6:+.2f}M to {d['high'] / 1e6:+.2f}M)"
        )

    dec.save_policy()
    with start_run("decision_policy", evaluated_on="validation", step=STEP):
        mlflow.log_params(
            {
                "monthly_fee_rate": config.MONTHLY_FEE_RATE,
                "monthly_funding_cost": config.MONTHLY_FUNDING_COST,
                "loss_given_default": config.LOSS_GIVEN_DEFAULT,
                **{f"threshold_{t}m": round(v, 4) for t, v in dec.thresholds_by_term().items()},
            }
        )
        mlflow.log_metrics(
            {
                "val_profit_per_1000_kes": val.loc["cost_rule", "profit_per_1000_applicants_kes"],
                "val_approval_rate": val.loc["cost_rule", "approval_rate"],
                "val_default_rate_approved": val.loc["cost_rule", "default_rate_approved"],
                "val_gain_vs_provisional_per_1000": vs_provisional["difference"],
                "cv_best_scale": best_scale,
                "cv_realised_to_expected": realised / expected,
            }
        )
        mlflow.log_text(policies.to_csv(), "decision_policies_cv.csv")
        mlflow.log_text(val.to_csv(), "decision_policies_validation.csv")
        mlflow.log_text(sens.to_csv(index=False), "decision_sensitivity.csv")
    print(
        f"Saved decision_policies.csv, decision_sensitivity.csv, "
        f"{config.DECISION_POLICY_FILE.name} and {CHART.name}"
    )


if __name__ == "__main__":
    main()
