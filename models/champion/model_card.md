# Model card: mkopoguard-champion v1.0.0

## What it does
Estimates the probability that a microloan applicant will default, and recommends approve or
decline using the lender's costs. It supports a loan officer's decision; it does not replace it.

- **Model:** logistic regression (L2, C = 1) inside a scikit-learn pipeline (log, clip, impute,
  scale, one-hot). Uncalibrated: its probabilities were already honest (step 3.10).
- **Inputs:** 32 columns in 5 groups: loan, profile, savings_resilience, borrowing_history, statement.
  Applicants without an M-Pesa statement ("thin-file") are scored from the survey answers.
- **Not used:** country, gender, or any identifier.
- **Trained on:** 15,910 synthetic applicants
  (training + validation).
- **MLflow run:** `76e2ee2952374816b4e9c673a4ab20fa`

## Intended use
- Portfolio and teaching project: a worked example of building and testing a credit model.
- Screening digital microloans of KSh 1,000-100,000 over 1, 3 or 6 months.

## Not intended for
- Real lending decisions. The applicants are **synthetic**: real Global Findex 2021 survey
  answers from 20 Sub-Saharan African countries, with loans, statements and defaults simulated.
- Loans outside the ranges above, or business / secured loans.
- Fully automatic declines without a human able to review and override them.

## Performance on the locked test set
Scored once, on 2,808 applicants the model never saw
(default rate 19.8%).
Brackets are 95% bootstrap intervals.

| measure | value |
|---|---|
| PR-AUC (primary) | 0.399 (0.360-0.443) |
| ROC-AUC | 0.717 (0.693-0.740) |
| KS | 0.338 (0.300-0.384) |
| Ceiling: the simulator's true probabilities | 0.619 |
| No-skill baseline (= default rate) | 0.198 |
| Calibration error (ECE) | 0.015 |
| Calibration slope / intercept | 0.93 / -0.08 |
| Average predicted default rate | 19.7% |
| Actual default rate | 19.8% |

| group | applicants | default rate | PR-AUC | ECE | approved |
|---|---|---|---|---|---|
| All applicants | 2,808 | 19.8% | 0.399 (0.360-0.443) | 0.015 | 49% |
| Kenya | 144 | 16.0% | 0.342 (0.208-0.542) | 0.053 | 50% |
| Statement holders | 1,476 | 16.6% | 0.412 (0.353-0.472) | 0.025 | 61% |
| Thin-file (no statement) | 1,332 | 23.3% | 0.385 (0.337-0.442) | 0.011 | 36% |

## Decision rule
Approve when the probability of default is below the threshold for the loan's term. The
thresholds come from the lender's costs: fee 7.5% a month, funding
cost 1.0% a month, loss given default
80% of principal (assumptions; see docs/evaluation_plan.md).

| loan term | decline when probability of default is at least |
|---|---|
| 1 month(s) | 0.075 |
| 3 month(s) | 0.196 |
| 6 month(s) | 0.328 |

On the test set the rule approved 49% of applicants, with
11.0% defaulting among those approved, and earned
KSh 0.87M more per 1,000 applicants than approving everyone
(95% interval 0.61M to 1.13M).

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
