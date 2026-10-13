# Evaluation plan

Written in step 3.2, **before any model was trained**. Every model in Phase 3 is scored by `evaluate()` in `src/mkopoguard/models/metrics.py`, and these rules decide which one wins.

## Metrics

| Metric | Measures | No skill | Ceiling* | Role |
| --- | --- | --- | --- | --- |
| **PR-AUC** | Finding defaulters without false alarms | 0.199 (= default rate) | 0.614 | **Primary: picks the model** |
| ROC-AUC | Ranking risky above safe | 0.500 | 0.846 | Reported; leakage alarm |
| KS | Largest gap between defaulter and non-defaulter scores | 0.000 | 0.543 | Reported (credit-industry standard) |
| Brier / Brier skill | Accuracy of the probabilities | 0.160 / 0.000 | 0.113 / 0.294 | Checks calibration (step 3.10) |
| ECE | Predicted vs actual default rate across 10 bins | 0.000† | 0.013 | Checks calibration |
| Recall, precision, false-positive rate, approval rate | What happens to applicants at the threshold | — | — | Reported at the threshold |
| Accuracy | Share of correct decisions | **0.801** | 0.772 | Reported, never used to choose |

\* *Ceiling* = scoring with each applicant's true hidden default probability, on the validation set. No real model can beat it.
† Guessing the average is perfectly calibrated but useless, which is why calibration is never judged alone.

## Rules

1. **Model choice:** highest PR-AUC in 5-fold cross-validation on the training set, confirmed on the validation set. A model must beat logistic regression to justify extra complexity.
2. **Threshold:** decision metrics use a provisional threshold of 0.20 (about the default rate) until step 3.11 picks one from the cost of each kind of mistake.
3. **Leakage alarm:** any ROC-AUC above 0.86 or PR-AUC above 0.65 is treated as a leak, not a success, and is investigated before anything else.
4. **Groups:** every result is also reported for Kenya, statement holders and thin-file applicants. Groups under 500 applicants (e.g. ~141 Kenyans in validation, ~144 in test) are reported with a 95% bootstrap confidence interval (1,000 resamples).
5. **Test set:** opened once, in step 3.12, after every choice is final. Its scores are logged with `evaluated_on=test` and never used to change the model.

## Why not accuracy?

80% of applicants repay, so approving everyone is 80% accurate while catching zero defaulters. On the validation set, the do-nothing model's accuracy (0.801) beats even perfect knowledge (0.772). Accuracy rewards ignoring the minority class, which is exactly the class a lender cares about.

## Rule added in step 3.3: ties

6. **A difference in CV PR-AUC smaller than one fold standard deviation is a tie, and ties go to the simpler model.** Added after the baselines showed three logistic variants within 0.002 of each other, against a fold spread of about 0.029. The rule doesn't favour any model in particular; it stops noise from picking winners.

## Baselines (step 3.3)

| Model | CV PR-AUC | Val PR-AUC | Val ROC-AUC | Val ECE |
| --- | --- | --- | --- | --- |
| **logistic (baseline to beat)** | **0.393 ± 0.029** | 0.412 | 0.717 | 0.011 |
| logistic + country | 0.394 ± 0.027 | 0.410 | 0.717 | 0.010 |
| logistic, class-weighted | 0.392 ± 0.029 | 0.411 | 0.717 | 0.252 |
| dummy | 0.199 | 0.199 | 0.500 | 0.000 |

Decisions:
- **country_code is dropped** for good: no gain overall (a tie), and worse for Kenyans (val PR-AUC 0.362 vs 0.390).
- **Class weights don't improve ranking but ruin calibration** (ECE 0.011 → 0.252; approval at the 0.20 threshold falls from 60% to 10%). Any imbalance method in step 3.5 must be followed by calibration (step 3.10) before its probabilities are used.

## Candidates (step 3.4)

| Model | CV PR-AUC | vs logistic | Train PR-AUC | Val PR-AUC | Overfit gap |
| --- | --- | --- | --- | --- | --- |
| **logistic** | **0.393** | — | 0.405 | 0.412 | −0.007 |
| XGBoost | 0.381 | −0.012 | 0.568 | 0.400 | 0.168 |
| Random forest | 0.378 | −0.014 | 0.535 | 0.381 | 0.154 |
| LightGBM | 0.378 | −0.015 | 0.628 | 0.395 | 0.233 |

**No tree model beats logistic regression.** Two reasons, both checked:
1. The trees overfit: training PR-AUC 0.54–0.63 vs validation 0.38–0.40 (see `docs/img/models_01_overfitting.png`).
2. The simulated default rule is a weighted sum passed through a logistic curve, with no interactions for trees to find. Given the true hidden drivers directly, logistic regression still beats XGBoost (validation PR-AUC 0.580 vs 0.566).

Carried into tuning (step 3.6): logistic regression and XGBoost (the best tree model), to test whether stronger regularisation closes the gap.

## Imbalance (step 3.5)

About 20% of applicants default (4.02 non-defaulters per defaulter in training). We compared
three ways of handling this for the two models carried forward from step 3.4. SMOTE ran inside
an imbalanced-learn pipeline, so only training folds were ever resampled (tested in
`tests/unit/test_imbalance.py`). Full results: `docs/results/imbalance.csv`.

| run | CV PR-AUC | gain vs none | val ECE | val approval rate @0.20 |
|---|---|---|---|---|
| logistic_none | 0.393 | – | 0.011 | 60% |
| logistic_weights | 0.392 | −0.000 | 0.252 | 10% |
| logistic_smote | 0.391 | −0.002 | 0.241 | 14% |
| xgboost_none | 0.381 | – | 0.013 | 60% |
| xgboost_weights | 0.378 | −0.002 | 0.227 | 15% |
| xgboost_smote | 0.375 | −0.006 | 0.034 | 55% |

**Finding:** no strategy beats "none" by more than one fold standard deviation (rule 6), so none
is worth it. Weighting shifts every score up by roughly the same amount, so the ranking (PR-AUC)
barely changes, but the probabilities stop meaning what they say: ECE rises about 20-fold and
almost every applicant lands above the 0.20 threshold. SMOTE's invented defaulters add noise,
not new patterns, which hurts XGBoost most.

**Decision:** train on the data as it is. The imbalance is handled where it belongs: calibration
(step 3.10) and a threshold chosen from the cost of each kind of mistake (step 3.11). Moving
the threshold buys recall without corrupting the probabilities a loan officer sees.

## Tuning (step 3.6)

Optuna (TPE sampler, seed 2026) searched 40 logistic and 60 XGBoost settings. Every trial was
scored by 5-fold CV on the training set only; trial 0 was always the untuned model. Results:
`docs/results/tuning.csv`, every trial: `docs/results/tuning_trials.csv`, chart:
`docs/img/models_02_tuning.png`.

| model | untuned CV PR-AUC | best tuned | gain | 1 fold std | val PR-AUC untuned → tuned | overfit gap untuned → tuned |
|---|---|---|---|---|---|---|
| logistic | 0.393 | 0.398 | +0.005 | 0.029 | 0.412 → 0.412 | −0.007 → −0.009 |
| xgboost | 0.381 | 0.397 | +0.017 | 0.023 | 0.400 → 0.408 | 0.168 → 0.002 |

**Findings**
- XGBoost's best settings use `max_depth = 1`: one feature per tree, i.e. an additive model.
  Given the choice, boosting turned itself into a version of logistic regression and its
  overfitting vanished. This confirms step 3.4: the default rule has no interactions to find.
- Logistic's best settings are almost pure lasso (l1_ratio ≈ 0.99, C ≈ 0.05), which zeroes 38
  of the 68 columns with no loss of score. Followed up in the ablation (step 3.7).
- Neither gain beats one fold std (rule 6), and the best of many trials is slightly
  optimistic anyway.

**Decision:** keep the untuned settings (`docs/results/tuned_params.json`). The champion is
plain logistic regression (L2, C = 1): as good as anything tuned, the simplest to explain to a
loan officer, and well calibrated. Later steps build models with `tuning.chosen_model()`.

## Learning curve (step 3.8)

Each CV fold's training rows were subsampled to 1/16, 1/8, 1/4, 1/2 and all (3 random draws
for each smaller share), and every model was scored on the same full held-out fold. Both
models use the 3.6 settings and the 3.7 feature groups. Results:
`docs/results/learning_curve.csv`, chart: `docs/img/models_04_learning_curve.png`.

| applicants trained on | logistic: unseen | logistic: training | XGBoost: unseen | XGBoost: training |
|---|---|---|---|---|
| 655 | 0.346 | 0.494 | 0.314 | 0.886 |
| 1,310 | 0.369 | 0.435 | 0.332 | 0.839 |
| 2,620 | 0.382 | 0.417 | 0.353 | 0.767 |
| 5,241 | 0.392 | 0.410 | 0.369 | 0.675 |
| 10,482 | 0.395 | 0.404 | 0.381 | 0.591 |

**Ceiling:** the simulator's true default probabilities score PR-AUC 0.617 on the same folds.
No model can beat this with any amount of data, because defaults are partly random. (The
leakage alarm at 0.65 sits just above it.) The true probabilities are read for evaluation only,
through load_split, so the test set stays locked.

**Findings**
- Logistic regression has plateaued: the last two doublings gained 0.010 and 0.003, and the next
  is projected to gain 0.001, below the 0.002 that counts. Its overfitting gap is 0.009.
- The 0.22 gap to the ceiling is information the inputs don't carry (true discipline, income,
  noise in behaviour), not a shortage of applicants. Closing it needs new kinds of data
  (e.g. longer statement history, repayment records), not more of the same.
- XGBoost is still learning (+0.011 per doubling, overfitting gap 0.21), but on the current
  trend it would need about 4x the applicants just to reach logistic's score. Step 3.6
  showed it only gets there by becoming additive, so more data would not reveal interactions.

**Decision:** the 18,718-applicant pool is large enough. Keep logistic regression; don't grow
the simulated pool.

## Leave-one-country-out (step 3.9)

**Rule 9.** The model transfers to new countries if (a) no country loses clearly when the model
has never seen it: a PR-AUC loss above 0.02 whose paired bootstrap interval (widened to
1 - 0.05/20 for checking 20 countries) lies entirely above zero; and (b) no never-seen
country's average prediction is off by more than chance (Bonferroni, |z| > 3.02).

**Design.** A first attempt (train on 19 countries, compare with ordinary CV) made every
country look better when unseen, because the 19-country model had ~12,400 training rows vs
~10,500 in a CV fold: it measured more data, not transfer. The final design is paired: in
each CV fold, each country's held-out applicants are scored by the fold's model and by the
same model retrained without that country (~5% fewer rows). Results:
`docs/results/country_transfer.csv`, `docs/results/country_transfer_calibration.csv`,
chart `docs/img/models_06_country_transfer.png`.

| | champion (logistic) | XGBoost |
|---|---|---|
| pooled PR-AUC, seen → never seen | 0.392 → 0.391 | 0.377 → 0.375 |
| median loss per country | −0.0005 | +0.0006 |
| worst country | ZWE +0.004 | KEN +0.025 |
| Kenya PR-AUC, seen → never seen | 0.433 → 0.435 | 0.403 → 0.378 |
| countries losing clearly | 0 | 0 |
| never-seen countries miscalibrated | 0 (largest ZWE, z = 2.96) | 1 (ZWE, z = 3.05) |
| rule 9 | transfers | does not transfer (narrowly) |

**Findings**
- The champion barely depends on which countries it was trained on; its per-country swings are
  about a third of XGBoost's. With no country input and an additive form, it has no room for
  country-specific shortcuts. This supports dropping `country_code` in step 3.3.
- XGBoost leans more on local patterns (Kenya −0.025, not significant) and narrowly fails the
  calibration check for Zimbabwe.
- Zimbabwe is a watch item: its default rate (27.4%) is above what its applicants' answers
  predict (22.8%) in 3.9 and 3.10. The simulator has no country effect, so this is most likely
  chance in 635 applicants; to be checked again in monitoring.

**Limitation.** In the real Findex, `income_quintile` is relative within each country, so the
same quintile means different incomes in different countries. The simulator gives each
quintile the same Kenya-equivalent income everywhere, so it cannot show that transfer risk.
A real deployment must re-check transfer on real repayment data.

**Decision:** the champion may be used in countries outside the training pool without
retraining, subject to calibration monitoring. Country stays out of the model.

## Calibration (step 3.10)

**Rule 8.** A calibration method is adopted only if, compared with none, it cuts ECE by more
than its paired fold standard error and by more than 0.005, without making the Brier score
worse or costing more than 0.002 PR-AUC.

Measured on out-of-fold CV predictions for all 13,102 training applicants. Methods: none,
sigmoid (Platt) and isotonic, each fitted with an inner 5-fold CV. A **positive control** (the
class-weighted logistic from 3.5, miscalibrated on purpose) shows the method finds and fixes
bad calibration when it exists. Results: `docs/results/calibration.csv`,
`docs/results/calibration_by_group.csv`, chart `docs/img/models_05_calibration.png`.

| model | method | PR-AUC | ECE | slope | intercept | mean predicted (actual 0.199) |
|---|---|---|---|---|---|---|
| champion | none | 0.392 | 0.006 | 0.96 | −0.05 | 0.199 |
| champion | sigmoid | 0.392 | 0.006 | 1.01 | 0.02 | 0.199 |
| champion | isotonic | 0.389 | 0.006 | 0.94 | −0.07 | 0.199 |
| class-weighted (control) | none | 0.391 | 0.253 | 0.94 | −1.38 | 0.452 |
| class-weighted (control) | sigmoid | 0.391 | 0.006 | 1.02 | 0.02 | 0.199 |
| XGBoost | none | 0.377 | 0.011 | 0.85 | −0.17 | 0.197 |
| XGBoost | isotonic | 0.379 | 0.008 | 0.99 | −0.02 | 0.199 |

**Findings**
- The champion is already calibrated; neither method passes rule 8 (isotonic slightly hurts
  ranking). Logistic regression fitted by maximum likelihood is calibrated by construction.
- The positive control works: Platt scaling takes the class-weighted model from 45% predicted
  vs 20% actual (ECE 0.253) to 19.9% (ECE 0.006), confirming the 3.5 decision.
- XGBoost is mildly over-confident (slope 0.85), typical of boosted trees.
- By country (calibration-in-the-large): Kenya predicted 17.4% vs actual 16.2% (z = −0.9).
  With the flag bar raised for 20 countries (Bonferroni, |z| > 3.02), no country is flagged;
  the largest is Zimbabwe (z = 2.5), within chance. Thin-file and statement applicants:
  observed/expected 1.001 each. The simulator has no country effect, so this is the known answer.
- Validation (once): ECE 0.012, slope 0.99, intercept −0.005, mean predicted 19.8% vs 19.9%.

**Decision:** no calibration (`docs/results/calibration.json`). Probabilities are used as the
model gives them; step 3.11 can set the threshold directly from costs.

## Decision threshold (step 3.11)

**Rule.** Approve when the probability of default p < m·T / (m·T + LGD), where m is the net
margin per month, T the loan term in months and LGD the share of principal lost on default.
This is where approving has zero expected profit; the loan amount cancels out.

**Assumptions** (all settings in `config.py`):

| assumption | value | basis |
|---|---|---|
| fee | 7.5% of principal per month | M-Shwari's 30-day price ([CGAP](https://www.cgap.org/blog/digital-credit-in-kenya-time-for-celebration-or-concern); [Business Daily, 2023](https://www.businessdailyafrica.com/bd/markets/market-news/m-shwari-retains-7-5pc-rate-as-competitors-increase-fee--4183742)); the cheap end of Kenyan digital credit (Tala 5-20% per loan, Branch 1-14%/month, Okolea 11.15%/30 days; [Kenya Times, 2024](https://thekenyatimes.com/business/interest-rates-for-top-licensed-mobile-loan-apps-in-kenya/)) |
| funding cost | 1% per month (~12% a year) | assumption; CBK rate 8.75% ([Aug 2026](https://khusoko.com/2026/08/11/cbk-holds-central-bank-rate-8-75-percent-august-2026/)) plus a lender's spread |
| loss given default | 80% of principal | assumption; unsecured, small loans with costly recovery (CBK: 83.1% of loans under KSh 1,000 default, 16.4% NPL for KSh 50k-100k; [Business Daily, 2025](https://www.businessdailyafrica.com/bd/economy/borrowers-fail-to-repay-83pc-of-digital-loans-below-sh1000-5189888)); Basel's unsecured corporate benchmark is 45% |

Net margin 6.5% a month gives thresholds of 0.075 (1-month loans), 0.196 (3-month) and
0.328 (6-month). Results: `docs/results/decision_policies.csv`,
`docs/results/decision_sensitivity.csv`, chart `docs/img/models_07_decision.png`.

| policy (out-of-fold CV) | approved | default rate among approved | profit per 1,000 applicants |
|---|---|---|---|
| approve everyone | 100% | 19.9% | KSh 0.85M |
| provisional 0.20 | 60% | 11.3% | KSh 1.40M |
| best single cut-off (0.21) | 62% | 11.7% | KSh 1.42M |
| cost-based rule | 49% | 12.0% | KSh 1.67M |

**Findings**
- The cost-based rule beats every single cut-off (+19% over the provisional 0.20).
- The thresholds are optimal in practice: scaling all three, profit peaks at 1.05x; expected
  profit (KSh 22.4M) matches realised (KSh 21.8M) within 2%. Both confirm the 3.10 calibration.
- Validation (once): +KSh 0.87M per 1,000 applicants vs approving everyone (95% interval
  +0.62M to +1.12M); +0.06M vs provisional 0.20, not significant on 2,808 applicants
  (-0.11M to +0.24M).
- Approval by term: 15% (1-month), 60% (3-month), 85% (6-month). A 1-month loan earns 6.5% if
  repaid but loses 80% on default, so only very safe borrowers qualify; short loans need a
  higher price or a longer term, not a looser rule.
- Thin-file applicants are approved 34% of the time vs 62% for statement holders: uploading a
  statement roughly doubles an applicant's chance (a design point for the app in Phase 5;
  fairness by gender is audited in a later phase).
- Sensitivity (net margin 3-10%/month x LGD 45-100%): the rule is profitable in all 16
  scenarios; approving everyone loses money in 6.

**Limitations.** The simulator's default rate does not depend on loan term (in practice longer
loans usually default more), and the profit model is simplified: flat fee on principal, no
partial repayment before default, no operating or acquisition costs.

**Decision:** use the cost-based rule (`docs/results/decision_policy.json`). The API calls
`decision.decide(p, term_months, policy)` for each applicant.

## Final test (step 3.12)

**Procedure.** The four decision files (3.6 settings, 3.7 feature groups, 3.10 calibration,
3.11 decision rule) were fingerprinted into the report; the code was committed first (the
script refuses uncommitted changes); the champion was refitted on training + validation
(15,910 applicants) and scored once on the 2,808 test applicants. Every opening of the test set
is logged in `docs/results/test_runs.log`; a second opening requires a written reason. Nothing
about the model was changed after the test. Results: `docs/results/test_report.json`,
`docs/results/test_by_group.csv`, chart `docs/img/models_08_final_test.png`.

| measure | test | 95% interval | validation (expected) |
|---|---|---|---|
| PR-AUC | 0.399 | 0.360-0.443 | 0.412 |
| ROC-AUC | 0.717 | 0.693-0.740 | 0.717 |
| KS | 0.338 | 0.300-0.384 | 0.337 |
| ceiling (true probabilities) | 0.619 | | |
| no-skill baseline | 0.198 | | |

Calibration: ECE 0.015, slope 0.93, intercept -0.08; mean predicted 19.7% vs actual 19.8%.
Cost-based rule: approves 49%; defaults among approved 11.0% (vs 19.8% overall); profit
KSh 1.77M per 1,000 applicants, +KSh 0.87M vs approving everyone (95% interval +0.61M to +1.13M).

| group | n | PR-AUC (95% interval) | approved |
|---|---|---|---|
| all | 2,808 | 0.399 (0.360-0.443) | 49% |
| Kenya | 144 | 0.342 (0.208-0.542) | 50% |
| statement | 1,476 | 0.412 (0.353-0.472) | 61% |
| thin-file | 1,332 | 0.385 (0.337-0.442) | 36% |

**Reading the result**
- Validation's prediction held: every group's validation PR-AUC lies inside its test interval.
  The test is slightly lower (0.399 vs 0.412), as expected once the luck in a validation score
  is removed.
- The model ranks at about twice the no-skill level; the gap to the 0.62 ceiling is information
  the inputs don't carry (3.8).
- Probabilities stay honest on new applicants. The leakage alarm did not fire.
- Kenya's test interval is too wide to be informative (144 applicants, ~23 defaulters); the
  leave-one-country-out result in 3.9 (667 Kenyans) is the better evidence for Kenya.

**These are the project's final model numbers.** Any later change to the model needs a new,
untouched test set.
