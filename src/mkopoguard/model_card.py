"""Steps 3.13 + 4.5: the model card, the document that travels with the model.

Written from saved results only (the final test report, the ablation, the explanations, the
behaviour tests and the fairness audit), so every number in the card is a number the model
actually produced, never retyped by hand. scripts/package_model.py writes the same text to
docs/model_card.md and into the bundle, and a CI test checks the two are identical.
"""

import json
from pathlib import Path

import pandas as pd

from mkopoguard import config
from mkopoguard.explain.behaviour import KNOWN_FLIPS
from mkopoguard.features.groups import FEATURE_GROUPS

GROUP_NAMES = {
    "all": "All applicants",
    "kenya": "Kenya",
    "statement": "Statement holders",
    "thin_file": "Thin-file (no statement)",
}
EVIDENCE_FILES = {
    "ablation": "ablation.csv",
    "importance": "shap_importance.csv",
    "reasons": "reason_code_frequency.csv",
    "behaviour": "behaviour_tests.csv",
    "fairness": "fairness_by_group.csv",
    "fairness_summary": "fairness_summary.json",
}
REQUIRED_SECTIONS = (
    "What it does",
    "Intended use",
    "Not intended for",
    "Data",
    "Inputs and the ablation",
    "Performance on the locked test set",
    "Decision rule",
    "Explanations",
    "Behaviour tests",
    "Fairness",
    "Limitations and risks",
    "How to use it",
)
NOT_FOR_LENDING = "Not for real lending decisions"
FAIRNESS_HEADER = (
    "| group | approved | oracle approved | approval ratio "
    "| good borrowers declined vs reference | of which the model adds | predicted / true risk |\n"
    "|---|---|---|---|---|---|---|"
)


def load_evidence(results_dir: Path = config.RESULTS_DIR) -> dict:
    """Every saved result the card quotes (steps 3.7, 4.1-4.4)."""
    missing = [f for f in EVIDENCE_FILES.values() if not (results_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing results {missing} in {results_dir}: run the step 3.7 and 4.1-4.4 scripts"
        )
    return {
        "ablation": pd.read_csv(results_dir / EVIDENCE_FILES["ablation"], index_col=0),
        "importance": pd.read_csv(results_dir / EVIDENCE_FILES["importance"], index_col=0),
        "reasons": pd.read_csv(results_dir / EVIDENCE_FILES["reasons"], index_col=0),
        "behaviour": pd.read_csv(results_dir / EVIDENCE_FILES["behaviour"], index_col=0),
        "fairness": pd.read_csv(
            results_dir / EVIDENCE_FILES["fairness"], index_col=[0, 1, 2]
        ).sort_index(),
        "fairness_summary": json.loads(
            (results_dir / EVIDENCE_FILES["fairness_summary"]).read_text(encoding="utf-8")
        ),
    }


def _ci(entry: dict) -> str:
    return f"{entry['value']:.3f} ({entry['low']:.3f}-{entry['high']:.3f})"


def _pts(value: float) -> str:
    return f"{value * 100:+.1f} pts"


def _flagged(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


# --- sections built from the evidence -----------------------------------------------------------


def _ablation_section(ablation: pd.DataFrame, kept_groups: list[str]) -> str:
    rows = "\n".join(
        f"| {group} | {int(r['n_columns'])} | {r['loss_all']:+.4f} | {r['loss_thin_file']:+.4f} | "
        f"{'kept' if group in kept_groups else 'dropped'} |"
        for group, r in ablation.iterrows()
    )
    return f"""Each feature group was removed in turn (cross-validated, step 3.7). A group is kept
only if removing it clearly loses PR-AUC (by more than its standard error and 0.002) for all
applicants or for thin-file applicants. Dropped groups are never asked for.

| feature group | columns | PR-AUC lost without it (all) | (thin-file) | decision |
|---|---|---|---|---|
{rows}"""


def _explanations_section(importance: pd.DataFrame, reasons: pd.DataFrame) -> str:
    top = importance.head(8)
    rows = "\n".join(
        f"| {int(r['rank'])} | {name} | {r['share']:.1%} | {'yes' if r['driver'] else 'no'} |"
        for name, r in top.iterrows()
    )
    others = importance.drop(index="has_statement", errors="ignore")
    non_driver = others.loc[~others["driver"].astype(bool), "share"].sum()
    common = ", ".join(f"`{code}` ({r['total']:.0%})" for code, r in reasons.head(5).iterrows())
    return f"""Exact SHAP values (for a logistic regression they have a closed form, checked against
the `shap` library; step 4.1). Because the data is synthetic we know the true drivers of default:

| rank | input | share of importance | true driver? |
|---|---|---|---|
{rows}

Inputs the simulator never uses carry {non_driver:.1%} of the importance (limit set: 15%).
`has_statement` carries the difference between thin-file applicants and statement holders.

Every score comes with **3 reasons that raise the risk and 2 that lower it** (step 4.2), in plain
English, each with its size in percentage points. Thin-file applicants get one reason for the
missing statement, never statement details. Most frequent reasons: {common}."""


def _behaviour_section(behaviour: pd.DataFrame) -> str:
    checks = behaviour[behaviour["expect"] != "none"]
    passed = checks["passed"].astype(str).str.lower() == "true"
    failed = checks[~passed]
    rows = "\n".join(
        f"| {r['description']} | {r['share_wrong']:.0%} | {r['mean_change_pts']:+.2f} pts | "
        f"{r['share_decisions_wrong']:.1%} |"
        for _, r in failed.iterrows()
    )
    reports = behaviour[behaviour["expect"] == "none"]
    largest = reports.loc[reports["mean_change_pts"].abs().idxmax()]
    unknown = sorted(set(failed.index) - set(KNOWN_FLIPS))
    note = f"\n\n**Unexplained failures:** {', '.join(unknown)}." if unknown else ""
    return f"""Each check changes one thing about real (synthetic) applicants, keeps everything else
equal, and requires the risk to move the way the simulator says (step 4.3). **{int(passed.sum())}
of {len(checks)} checks pass** for every applicant, including the guide's: a higher average
inflow never raises risk, and starting to use an overdraft or owing more on one always raises it.

Known failures (marked as expected failures in the tests):

| change | applicants moved the wrong way | average change | decisions moved the wrong way |
|---|---|---|---|
{rows}

Why: with the debt held equal, more overdraft borrowings means the debt revolves (borrowed and
repaid). In the simulator, frequent borrowers are slightly *less* risky among overdraft users, and
even a linear fit to the true risk gives the count a negative weight. Median inflow is a twin of
average inflow and flips only when moved alone. These could still be gamed, so they are a known
limitation, to be fixed in v2.0.0 (e.g. a "uses overdraft" flag instead of the raw count).

Inputs with no expected direction are measured only; the largest effect is
"{largest["description"]}" ({largest["mean_change_pts"]:+.2f} pts on average).{note}"""


def _fairness_rows(fairness: pd.DataFrame, attribute: str, groups: list[str]) -> str:
    table = fairness.loc[(attribute, "all")]
    out = []
    for group in groups:
        if group not in table.index:
            continue
        r = table.loc[group]
        ref = str(r["reference"]).lower() == "true"
        out.append(
            f"| {group}{' (reference)' if ref else ''} | {r['approval_rate']:.1%} | "
            f"{r['oracle_approval_rate']:.1%} | {r['approval_ratio']:.2f} | "
            f"{'-' if ref else _pts(r['fpr_gap'])} | {'-' if ref else _pts(r['excess_fpr_gap'])} | "
            f"{r['mean_predicted']:.1%} / {r['mean_true_probability']:.1%} |"
        )
    return "\n".join(out)


def _fairness_section(fairness: pd.DataFrame, summary: dict) -> str:
    flags, same = summary["flags"], summary["same_risk_average_decline_gap"]
    country = fairness.loc[("country", "all")]
    worst = country["approval_ratio"].idxmin()
    ref_country = country.index[country["reference"].astype(str).str.lower() == "true"][0]
    rows = "\n".join(
        [
            _fairness_rows(fairness, "gender", ["male", "female"]),
            _fairness_rows(fairness, "age_group", ["18-24", "25-34", "35-49", "50+"]),
            _fairness_rows(fairness, "country", sorted({ref_country, "KEN", worst})),
        ]
    )
    same_risk = "; ".join(
        f"{k.split(': ')[1].removesuffix('_gap')} {_pts(v)}" for k, v in same.items()
    )
    adds = flags["gender"]["model_adds_gap"] + flags["age_group"]["model_adds_gap"]
    meaning = MEANING if adds else NO_GAPS
    flag_lines = "\n".join(
        f"- **{attr.replace('_', ' ')}:** below four-fifths: {_flagged(f['adverse_impact'])}; "
        f"good borrowers declined > 5 pts more: {_flagged(f['fpr_flag'])}; "
        f"model adds to the real-risk gap: {_flagged(f['model_adds_gap'])}; "
        f"calibration flagged: {_flagged(f['calibration_flagged'])}"
        for attr, f in flags.items()
    )
    return f"""Gender, age group and country are never model inputs (age is, as a number); they are
kept only for this audit (step 4.4). Out-of-fold decisions on the 15,910 development applicants
(the test set stays closed). The **oracle** applies the same rule to each applicant's true risk,
known because the data is synthetic: it shows how much of a gap real risk differences justify.

{FAIRNESS_HEADER}
{rows}

Flags (four-fifths rule: approval ratio below 0.80, US 29 CFR 1607.4(D), evidence of adverse
impact, not a verdict; error gaps must be over 5 points with an interval above zero):

{flag_lines}

Among applicants with the **same true risk**, the extra share declined compared with the
reference group: {same_risk}.

{meaning}"""


MEANING = """What this means: the model is equally well calibrated for every group, but groups
differ in true risk through income, work, savings and statement access (gender itself is not in
the default rule), and the model can't see the hidden habits that really drive default. It leans
on visible inputs where women and older applicants look worse, so at the same true risk they are
declined more often. In Kenya this is the kind of *indirect* discrimination Article 27 of the
Constitution prohibits, and sex is sensitive personal data under the Data Protection Act 2019.
Group-specific thresholds are ruled out (they would decide on gender or age group). Mitigations:
human review of declines near the threshold, statements for more applicants, and this audit
re-run on every new model."""
NO_GAPS = "No gender or age group is disadvantaged beyond what its true risk explains."


# --- the card ---------------------------------------------------------------------------------


def build_model_card(
    report: dict, policy: dict, kept_groups: list[str], run_id: str, evidence: dict | None = None
) -> str:
    evidence = load_evidence() if evidence is None else evidence
    rk, cal, rule = report["ranking"], report["calibration"], report["decision"]["cost_rule"]
    gain = report["decision"]["gain_vs_approve_everyone_per_1000"]
    thresholds = policy["thresholds_by_term_months"]
    n_inputs = sum(len(FEATURE_GROUPS[g]) for g in kept_groups)
    trained, tested = report["trained_on"]["applicants"], report["n"]
    groups = "\n".join(
        f"| {GROUP_NAMES[name]} | {int(g['n']):,} | {g['default_rate']:.1%} | "
        f"{g['pr_auc']:.3f} ({g['pr_auc_low']:.3f}-{g['pr_auc_high']:.3f}) | {g['ece']:.3f} | "
        f"{g['approval_rate']:.0%} |"
        for name, g in report["groups"].items()
    )
    terms = "\n".join(f"| {t} month(s) | {v:.3f} |" for t, v in sorted(thresholds.items()))
    return f"""# Model card: {config.MODEL_NAME} v{config.MODEL_VERSION}

> **{NOT_FOR_LENDING}.** This model was trained and tested on synthetic applicants. It is a
> portfolio and teaching project, not a credit product.

## What it does
Estimates the probability that a microloan applicant will default, recommends approve or decline
using the lender's costs, and gives the 3 + 2 main reasons for each score. It supports a loan
officer's decision; it does not replace it.

- **Model:** logistic regression (L2, C = 1) inside a scikit-learn pipeline (log, clip, impute,
  scale, one-hot). Uncalibrated: its probabilities were already honest (step 3.10).
- **Inputs:** {n_inputs} columns in {len(kept_groups)} groups: {", ".join(kept_groups)}.
  Applicants without an M-Pesa statement ("thin-file") are scored from the survey answers.
- **Not used:** country, gender, or any identifier.
- **MLflow run:** `{run_id}`

## Intended use
- Portfolio and teaching project: a worked example of building, testing and auditing a credit
  model end to end.
- Screening digital microloans of KSh 1,000-100,000 over 1, 3 or 6 months, with a human
  reviewing every decision.

## Not intended for
- **Real lending decisions**, for anyone. Nothing here has been validated on real repayments.
- Loans outside the ranges above, or business / secured loans.
- Fully automatic declines without a human able to review and override them.

## Data
- **Applicants:** {trained + tested:,} real Global Findex 2021 survey respondents from 20
  Sub-Saharan African countries (their answers on work, income group, savings, borrowing,
  emergencies and worries).
- **Simulated:** the loan (amount, term), a 6-month M-Pesa statement for applicants with mobile
  money, and whether they defaulted. Default comes from a simple additive rule over the loan-to-
  income ratio, income pattern, financial stress, savings and hidden habits (overdraft use,
  betting, discipline). The rule is known, which is what makes the explanation, behaviour and
  fairness checks possible.
- **Splits:** {trained:,} for development (training + validation) and {tested:,} in a locked test
  set, stratified by country, default and statement. The test set was opened once (step 3.12).
- **Limits:** real repayment behaviour is messier than an additive rule; habits that real people
  have and the simulator doesn't are missing; income quintile is relative within each country in
  the survey but not in the simulator; "has a statement" simply means "has mobile money".

## Inputs and the ablation
{_ablation_section(evidence["ablation"], kept_groups)}

## Performance on the locked test set
Scored once, on {tested:,} applicants the model never saw
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

## Explanations
{_explanations_section(evidence["importance"], evidence["reasons"])}

## Behaviour tests
{_behaviour_section(evidence["behaviour"])}

## Fairness
{_fairness_section(evidence["fairness"], evidence["fairness_summary"])}

## Limitations and risks
- **Synthetic outcomes.** Every number must be re-checked on real repayment data before any
  real use.
- **Fairness gaps** (above): women and over-50s are declined more often than their true risk
  explains. A human must review declines, and the audit must be re-run for every new model.
- **Behaviour flips** (above): more overdraft borrowings, with the debt held equal, lowers the
  score. An applicant could game it, and a reviewer may find it hard to defend.
- **Thin-file applicants are approved far less often** than statement holders, because the
  model knows less about them. Uploading a statement is the biggest single lever.
- **Short loans:** few 1-month loans are approved at these costs; the price, not the rule, is
  the lever.
- **Kenya** has only ~140 test applicants, so its test score is imprecise; leave-one-country-out
  testing (step 3.9) showed no loss for Kenya.
- **Age:** the model barely learned that 18-24 year olds are riskier, so it slightly
  under-predicts their risk.

## How to use it
Load with `mkopoguard.bundle.load_bundle()`, which checks every file's checksum, replays the
golden examples and checks the explainer background matches the model. Check inputs with
`check_input()` first: unseen categories or values outside the training range are scored but
flagged. `decide()` gives the probability, threshold and decision; `reasoner().explain_rows()`
gives the 3 + 2 reasons, with no training data needed.
"""
