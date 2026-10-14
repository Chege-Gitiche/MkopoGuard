# Model card: mkopoguard-champion v1.1.0

> **Not for real lending decisions.** This model was trained and tested on synthetic applicants. It is a
> portfolio and teaching project, not a credit product.

## What it does
Estimates the probability that a microloan applicant will default, recommends approve or decline
using the lender's costs, and gives the 3 + 2 main reasons for each score. It supports a loan
officer's decision; it does not replace it.

- **Model:** logistic regression (L2, C = 1) inside a scikit-learn pipeline (log, clip, impute,
  scale, one-hot). Uncalibrated: its probabilities were already honest (step 3.10).
- **Inputs:** 32 columns in 5 groups: loan, profile, savings_resilience, borrowing_history, statement.
  Applicants without an M-Pesa statement ("thin-file") are scored from the survey answers.
- **Not used:** country, gender, or any identifier.
- **MLflow run:** `76e2ee2952374816b4e9c673a4ab20fa`

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
- **Applicants:** 18,718 real Global Findex 2021 survey respondents from 20
  Sub-Saharan African countries (their answers on work, income group, savings, borrowing,
  emergencies and worries).
- **Simulated:** the loan (amount, term), a 6-month M-Pesa statement for applicants with mobile
  money, and whether they defaulted. Default comes from a simple additive rule over the loan-to-
  income ratio, income pattern, financial stress, savings and hidden habits (overdraft use,
  betting, discipline). The rule is known, which is what makes the explanation, behaviour and
  fairness checks possible.
- **Splits:** 15,910 for development (training + validation) and 2,808 in a locked test
  set, stratified by country, default and statement. The test set was opened once (step 3.12).
- **Limits:** real repayment behaviour is messier than an additive rule; habits that real people
  have and the simulator doesn't are missing; income quintile is relative within each country in
  the survey but not in the simulator; "has a statement" simply means "has mobile money".

## Inputs and the ablation
Each feature group was removed in turn (cross-validated, step 3.7). A group is kept
only if removing it clearly loses PR-AUC (by more than its standard error and 0.002) for all
applicants or for thin-file applicants. Dropped groups are never asked for.

| feature group | columns | PR-AUC lost without it (all) | (thin-file) | decision |
|---|---|---|---|---|
| loan | 2 | +0.0011 | +0.0025 | kept |
| profile | 4 | +0.0035 | +0.0092 | kept |
| financial_access | 5 | -0.0014 | -0.0008 | dropped |
| income_channels | 3 | -0.0003 | -0.0007 | dropped |
| savings_resilience | 4 | +0.0467 | +0.0712 | kept |
| borrowing_history | 4 | +0.0030 | +0.0012 | kept |
| transfers | 3 | -0.0010 | +0.0002 | dropped |
| statement | 18 | +0.0545 | +0.0005 | kept |

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

## Explanations
Exact SHAP values (for a logistic regression they have a closed form, checked against
the `shap` library; step 4.1). Because the data is synthetic we know the true drivers of default:

| rank | input | share of importance | true driver? |
|---|---|---|---|
| 1 | saved_past_year | 10.8% | yes |
| 2 | has_statement | 10.6% | no |
| 3 | stmt_betting_share | 10.0% | yes |
| 4 | worried_about_bills | 9.2% | yes |
| 5 | in_workforce | 5.8% | yes |
| 6 | loan_amount_kes | 5.0% | yes |
| 7 | income_quintile | 5.0% | yes |
| 8 | borrowed_for_medical | 5.0% | yes |

Inputs the simulator never uses carry 8.0% of the importance (limit set: 15%).
`has_statement` carries the difference between thin-file applicants and statement holders.

Every score comes with **3 reasons that raise the risk and 2 that lower it** (step 4.2), in plain
English, each with its size in percentage points. Thin-file applicants get one reason for the
missing statement, never statement details. Most frequent reasons: `saved_past_year` (88%), `bill_worry` (78%), `betting` (51%), `loan_amount` (50%), `work_status` (44%).

## Behaviour tests
Each check changes one thing about real (synthetic) applicants, keeps everything else
equal, and requires the risk to move the way the simulator says (step 4.3). **27
of 30 checks pass** for every applicant, including the guide's: a higher average
inflow never raises risk, and starting to use an overdraft or owing more on one always raises it.

Known failures (marked as expected failures in the tests):

| change | applicants moved the wrong way | average change | decisions moved the wrong way |
|---|---|---|---|
| 10 more overdraft borrowings, same amount owed | 99% | -0.84 pts | 2.2% |
| Median monthly inflow x 1.5 | 93% | +0.27 pts | 0.7% |
| 10 more overdraft borrowings and +0.1 months of income owed | 95% | -0.43 pts | 0.7% |

Why: with the debt held equal, more overdraft borrowings means the debt revolves (borrowed and
repaid). In the simulator, frequent borrowers are slightly *less* risky among overdraft users, and
even a linear fit to the true risk gives the count a negative weight. Median inflow is a twin of
average inflow and flips only when moved alone. These could still be gamed, so they are a known
limitation, to be fixed in v2.0.0 (e.g. a "uses overdraft" flag instead of the raw count).

Inputs with no expected direction are measured only; the largest effect is
"Borrowed from fin institution: yes vs no (non-driver)" (-1.41 pts on average).

## Fairness
Gender, age group and country are never model inputs (age is, as a number); they are
kept only for this audit (step 4.4). Out-of-fold decisions on the 15,910 development applicants
(the test set stays closed). The **oracle** applies the same rule to each applicant's true risk,
known because the data is synthetic: it shows how much of a gap real risk differences justify.

| group | approved | oracle approved | approval ratio | good borrowers declined vs reference | of which the model adds | predicted / true risk |
|---|---|---|---|---|---|---|
| male (reference) | 53.4% | 62.1% | 1.00 | - | - | 18.6% / 18.6% |
| female | 44.0% | 55.4% | 0.82 | +9.0 pts | +4.1 pts | 21.0% / 21.2% |
| 18-24 | 47.7% | 55.1% | 0.91 | +4.9 pts | -2.0 pts | 20.0% / 21.6% |
| 25-34 (reference) | 52.5% | 63.5% | 1.00 | - | - | 18.8% / 18.1% |
| 35-49 | 49.7% | 59.3% | 0.95 | +2.7 pts | -0.9 pts | 19.4% / 19.3% |
| 50+ | 38.8% | 52.2% | 0.74 | +14.0 pts | +4.2 pts | 22.9% / 22.8% |
| GAB (reference) | 59.3% | 66.4% | 1.00 | - | - | 15.5% / 15.2% |
| KEN | 59.2% | 65.5% | 1.00 | -2.7 pts | -1.4 pts | 17.2% / 17.4% |
| LSO | 34.9% | 47.5% | 0.59 | +23.2 pts | +7.9 pts | 24.4% / 25.0% |

Flags (four-fifths rule: approval ratio below 0.80, US 29 CFR 1607.4(D), evidence of adverse
impact, not a verdict; error gaps must be over 5 points with an interval above zero):

- **gender:** below four-fifths: none; good borrowers declined > 5 pts more: female; model adds to the real-risk gap: female; calibration flagged: none
- **age group:** below four-fifths: 50+; good borrowers declined > 5 pts more: 50+; model adds to the real-risk gap: 50+; calibration flagged: none
- **country:** below four-fifths: BWA, COG, LBR, LSO, MOZ, NAM, SWZ, ZMB, ZWE; good borrowers declined > 5 pts more: BEN, BWA, CMR, COG, LBR, LSO, MOZ, NAM, SWZ, ZMB, ZWE; model adds to the real-risk gap: BEN, COG, LBR, LSO, ZMB; calibration flagged: none

Among applicants with the **same true risk**, the extra share declined compared with the
reference group: female +6.9 pts; 18-24 +0.9 pts; 35-49 +1.6 pts; 50+ +9.1 pts.

What this means: the model is equally well calibrated for every group, but groups
differ in true risk through income, work, savings and statement access (gender itself is not in
the default rule), and the model can't see the hidden habits that really drive default. It leans
on visible inputs where women and older applicants look worse, so at the same true risk they are
declined more often. In Kenya this is the kind of *indirect* discrimination Article 27 of the
Constitution prohibits, and sex is sensitive personal data under the Data Protection Act 2019.
Group-specific thresholds are ruled out (they would decide on gender or age group). Mitigations:
human review of declines near the threshold, statements for more applicants, and this audit
re-run on every new model.

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
