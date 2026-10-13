"""Step 3.13: the model card, a one-page summary that travels with the model.

Written from the final test report (3.12) and the frozen decisions, so the numbers in the
card are always the numbers the model actually achieved, never retyped by hand.
"""

from mkopoguard import config
from mkopoguard.features.groups import FEATURE_GROUPS

GROUP_NAMES = {
    "all": "All applicants",
    "kenya": "Kenya",
    "statement": "Statement holders",
    "thin_file": "Thin-file (no statement)",
}


def _ci(entry: dict) -> str:
    return f"{entry['value']:.3f} ({entry['low']:.3f}-{entry['high']:.3f})"


def build_model_card(report: dict, policy: dict, kept_groups: list[str], run_id: str) -> str:
    rk, cal, rule = report["ranking"], report["calibration"], report["decision"]["cost_rule"]
    gain = report["decision"]["gain_vs_approve_everyone_per_1000"]
    thresholds = policy["thresholds_by_term_months"]
    n_inputs = sum(len(FEATURE_GROUPS[g]) for g in kept_groups)
    groups = "\n".join(
        f"| {GROUP_NAMES[name]} | {int(g['n']):,} | {g['default_rate']:.1%} | "
        f"{g['pr_auc']:.3f} ({g['pr_auc_low']:.3f}-{g['pr_auc_high']:.3f}) | {g['ece']:.3f} | "
        f"{g['approval_rate']:.0%} |"
        for name, g in report["groups"].items()
    )
    terms = "\n".join(f"| {t} month(s) | {v:.3f} |" for t, v in sorted(thresholds.items()))
    return f"""# Model card: {config.MODEL_NAME} v{config.MODEL_VERSION}

## What it does
Estimates the probability that a microloan applicant will default, and recommends approve or
decline using the lender's costs. It supports a loan officer's decision; it does not replace it.

- **Model:** logistic regression (L2, C = 1) inside a scikit-learn pipeline (log, clip, impute,
  scale, one-hot). Uncalibrated: its probabilities were already honest (step 3.10).
- **Inputs:** {n_inputs} columns in {len(kept_groups)} groups: {", ".join(kept_groups)}.
  Applicants without an M-Pesa statement ("thin-file") are scored from the survey answers.
- **Not used:** country, gender, or any identifier.
- **Trained on:** {report["trained_on"]["applicants"]:,} synthetic applicants
  (training + validation).
- **MLflow run:** `{run_id}`

## Intended use
- Portfolio and teaching project: a worked example of building and testing a credit model.
- Screening digital microloans of KSh 1,000-100,000 over 1, 3 or 6 months.

## Not intended for
- Real lending decisions. The applicants are **synthetic**: real Global Findex 2021 survey
  answers from 20 Sub-Saharan African countries, with loans, statements and defaults simulated.
- Loans outside the ranges above, or business / secured loans.
- Fully automatic declines without a human able to review and override them.

## Performance on the locked test set
Scored once, on {report["n"]:,} applicants the model never saw
(default rate {report["default_rate"]:.1%}).
Brackets are 95% bootstrap intervals.

| measure | value |
|---|---|
| PR-AUC (primary) | {_ci(rk["pr_auc"])} |
| ROC-AUC | {_ci(rk["roc_auc"])} |
| KS | {_ci(rk["ks"])} |
| Ceiling: the simulator's true probabilities | {report["ceiling_pr_auc"]:.3f} |
| No-skill baseline (= default rate) | {report["default_rate"]:.3f} |
| Calibration error (ECE) | {cal["ece"]:.3f} |
| Calibration slope / intercept | {cal["slope"]:.2f} / {cal["intercept"]:.2f} |
| Average predicted default rate | {cal["mean_predicted"]:.1%} |
| Actual default rate | {report["default_rate"]:.1%} |

| group | applicants | default rate | PR-AUC | ECE | approved |
|---|---|---|---|---|---|
{groups}

## Decision rule
Approve when the probability of default is below the threshold for the loan's term. The
thresholds come from the lender's costs: fee {policy["monthly_fee_rate"]:.1%} a month, funding
cost {policy["monthly_funding_cost"]:.1%} a month, loss given default
{policy["loss_given_default"]:.0%} of principal (assumptions; see docs/evaluation_plan.md).

| loan term | decline when probability of default is at least |
|---|---|
{terms}

On the test set the rule approved {rule["approval_rate"]:.0%} of applicants, with
{rule["default_rate_approved"]:.1%} defaulting among those approved, and earned
KSh {gain["difference"] / 1e6:.2f}M more per 1,000 applicants than approving everyone
(95% interval {gain["low"] / 1e6:.2f}M to {gain["high"] / 1e6:.2f}M).

## Limitations and risks
- **Synthetic outcomes.** The default rule is simple and additive; real repayment behaviour is
  messier. All numbers must be re-checked on real repayment data before any real use.
- **Thin-file applicants are approved about half as often** as statement holders, because the
  model knows less about them. Uploading a statement roughly doubles an applicant's chance.
- **Short loans:** few 1-month loans are approved at these costs; the price, not the rule, is
  the lever.
- **Kenya** has only ~140 test applicants, so its test score is imprecise; leave-one-country-out
  testing (step 3.9) showed no loss for Kenya.
- **Income quintile** is relative within each country in the real survey but not in the
  simulator, so cross-country transfer may be weaker in practice.
- **Fairness by gender** has not yet been audited (planned for a later phase).

## How to use it
Load with `mkopoguard.bundle.load_bundle()`, which checks every file's checksum and replays the
golden examples. Check inputs with `check_input()` first; unseen categories or values outside
the training range are scored but flagged.
"""
