# Data card: MkopoGuard synthetic applicants

## 1. What this dataset is

- **18,718 applicants**, one per adult (18+) respondent of the Global Findex 2021 survey in 20 Sub-Saharan countries where at least 40% of respondents have a mobile money account.
- **Real:** each applicant's survey answers (23 features, see `findex_columns.md`).
- **Simulated:** each applicant's loan, a 6-month mobile money statement, and whether they repaid.
- This is **not real lending data**. A model trained on it learns the simulator's rules below, not real-world truth. **Not for real lending decisions.**

## 2. Simulation design

All numbers below are starting assumptions, checked and tuned in step 2.2 (EDA on the synthetic data).

### 2.1 Timeline

- Each applicant gets an application date `T`, drawn at random within one year.
- The statement covers the 180 days before `T`. The loan is paid out at `T`; the outcome is known at `T + term`.
- **No transaction is ever generated on or after `T`.** This prevents data leakage and is enforced by a test.

### 2.2 Hidden traits (never given to the model)

| Trait | How it's drawn | Why it exists |
| --- | --- | --- |
| `monthly_income_kes` | Log-normal around a median set by income quintile (table below), × 0.5 if out of the workforce, spread σ = 0.5 | Sets the size of every transaction and the loan |
| `income_pattern` | **salaried** if `wage_payment_channel` is account or cash_only; else **seasonal** if `agri_payment_channel` is not none; else **irregular** | Shapes how money comes in |
| `discipline` | Standard normal, independent of the survey | An unobserved trait. It affects overdraft repayment, betting and default, so the statement reveals information the survey can't |
| `stress` | Built from `emergency_fund_source = could_not`, `worried_about_bills = very`, `borrowed_for_medical = 1`, `saved_past_year = 0`, plus noise | Financial fragility |

**Median monthly income by quintile (KES, assumption):**

| Quintile | 1 (poorest) | 2 | 3 | 4 | 5 (richest) |
| --- | --- | --- | --- | --- | --- |
| Median | 4,000 | 7,000 | 11,000 | 18,000 | 32,000 |


### 2.3 Loan

- `loan_to_income` ratio: log-normal with median 1.0, clipped to 0.3–3.0.
- `loan_amount_kes` = ratio × monthly income, rounded to the nearest KES 500 and clipped to KES 1,000–100,000.
- `term_months` ∈ {1, 3, 6}, with probabilities by amount:

| Amount (KES) | 1 month | 3 months | 6 months |
| --- | --- | --- | --- |
| ≤ 10,000 | 0.60 | 0.30 | 0.10 |
| 10,001–50,000 | 0.20 | 0.50 | 0.30 |
| > 50,000 | 0.05 | 0.35 | 0.60 |

**Decision: country income levels.** Income quintiles are measured *within* each country. Every country's quintiles are treated as Kenya-equivalent: the same income table applies everywhere, with no country multiplier. Reason: the app scores Kenyan applicants, and a multiplier would add an external data source for little benefit."

### 2.4 Mobile money statement

**Decision: applicants without mobile money.** Only applicants with `has_mobile_money = 1` get a statement. The rest get `has_statement = 0` and missing transaction features: they are thin-file borrowers, the first-time and informal applicants the original proposal targets, and the model must judge them from survey answers alone. Their hidden traits still exist and still feed the default rule; they just leave no statement trace."

Each transaction row: `applicant_id`, `timestamp`, `type`, `direction`, `amount_kes`, `balance_after`.

| Type | Direction | Who, and how often | Amount |
| --- | --- | --- | --- |
| `income_in` | in | salaried: 1/month; seasonal: 1–2 large lumps per quarter; irregular: 8–20 small inflows/month | Adds up to roughly monthly income (salaried ±5%, irregular ±40%) |
| `p2p_in` | in | If `received_domestic_remittance = 1`: about 1/month | 5–20% of income |
| `p2p_out` | out | If `sent_domestic_remittance = 1`: 1–3/month | 5–15% of income |
| `paybill` | out | Utilities; monthly if `utility_payment_channel = account`; missed months more likely with low discipline | 3–10% of income |
| `till` | out | If `made_digital_merchant_payment = 1`: 5–30/month | Small purchases |
| `airtime` | out | Everyone: 4–15/month | KES 20–200 |
| `overdraft_borrow` / `overdraft_repay` | in / out | Frequency rises with stress and low discipline; share repaid falls with low discipline | KES 100–5,000 |
| `betting` | out | Chance of being a bettor rises with low discipline; bettors spend 2–30% of outflows on it | Varies |

**Balance rule:** the balance never goes below zero. An outflow that would overdraw the account triggers an overdraft borrow if the applicant uses overdrafts, and is skipped otherwise.

### 2.5 Default rule

risk = b0 + Σ (weight × driver) + 0.8 × (−discipline) + noise, noise ~ Normal(0, 0.5)
p(default) = 1 / (1 + e^(−risk))
defaulted ~ Bernoulli(p(default))


`b0` is tuned so the overall default rate is about **20%** (allowed band 15–25%, in line with the CBK and Lendsqr figures cited in the original proposal).

| Driver | Comes from | Effect on risk |
| --- | --- | --- |
| Loan-to-income ratio | Loan | ↑ |
| Income pattern (irregular > seasonal > salaried) | Hidden | ↑ |
| Stress | Hidden | ↑ |
| True overdraft dependence | Hidden | ↑ |
| True betting share | Hidden | ↑ |
| `saved_in_savings_club` | Survey | ↓ |
| `saved_past_year` | Survey | ↓ |
| `in_workforce` | Survey | ↓ |
| Age 18–24 | Survey | ↑ (small) |

- **Never used in the default rule:** `is_female`, `country_code`. They're checked for fairness in Phase 4.
- The rule uses the **true hidden values**. The model only sees noisy versions computed from the statement, so it can't be perfect. A realistic target is ROC-AUC around 0.70–0.85, typical of real credit scorecards. Much higher would mean the simulator is too easy.
- Exact weights live in the simulator's code and are copied into this table once chosen.

### 2.6 Reproducibility

One seed, `SIMULATION_SEED = 42`, in `config.py`. The same seed must produce identical files (step 1.5 test).

## 3. Known limitations

- Survey answers are real; everything about loans, transactions and repayment is invented by the rules above.
- Survey answers are self-reported and from 2021–2022.
- The income table and transaction frequencies are informed assumptions, not measured values.
- The model's performance shows how well it recovers these rules. It is not evidence of real-world accuracy.
Incomes ignore differences between countries: a quintile-3 earner in a richer country (e.g. South Africa) is simulated with the same income as one in Kenya.
- Thin-file applicants (47%) are judged on survey answers only, so expect the model to be less accurate for them. Phase 3 reports results for both groups separately.

## 4. Final default-rule weights

Set in `src/mkopoguard/data/defaults.py`. Noise: Normal(0, 0.5). `b0` is solved automatically so the average default probability is 20% (on the generated data: b0 = -2.704).

| Driver | Weight |
| --- | --- |
| log(loan-to-income ratio) | +0.6 |
| Income pattern: seasonal (vs salaried) | +0.3 |
| Income pattern: irregular (vs salaried) | +0.5 |
| Stress | +0.5 |
| True overdraft propensity (0-1) | +1.5 |
| True betting share of income (0-0.3) | +4.0 |
| Low discipline (= -discipline) | +0.8 |
| Saved in a savings club | -0.4 |
| Saved in the past year | -0.3 |
| In the workforce | -0.3 |
| Age 18-24 | +0.2 |

**Results on the generated data:** default rate 19.9% (Kenya 16.1%); ceiling ROC-AUC using the true probabilities 0.84.

**Fairness note:** women default at 21.2% vs 18.3% for men even though gender is never used (a test proves it). The gap comes through correlated features such as income and employment. Phase 4 audits the model for it.
