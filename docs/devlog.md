# MkopoGuard v2 — Devlog

## 2026-10-05 — Phase 0 setup

**Did:** Created the repo, virtual environment, folder structure, Ruff + pre-commit, a smoke test and the GitHub Actions CI pipeline.

**Broke / learned:** I learnt how to create tests and create a github actions CI pipeline . I made a mistake in the creation of the __init__py files which I was able to resolve by finding the position of the files . I also had an intial conflict in pushing the README file to github after configuring the github actions pipeline but I was able to solve it in good time

**Next:** Phase 1.1 — audit notebook for the 20-country Findex pool.


## 2026-10-06 — Phase 1 Step 1.1 Auditing the dataset


**Did:** I was able to import the dataset to the project and audit it to find the required countries and columns I will be using for this project . I was able to perform tests on the dataset config file to ensure it has been appropriately set.

**Broke / learned:**
Learnt that about the package named as pyarrow which is used for interoperability between packages in python (numpy, pandas and others )
Learnt how to succesfully run notebooks on my machine
Used ISO country codes instead of names because "Côte d'Ivoire" gets mangled

**Decisions:**
- Choices I made and why (or "None")

## 2026-10-06 — Step 1.2: Decode the Findex columns + adults-only decision

**Goal:** Turn the survey's coded column names into plain-English features, and decide which columns the project keeps.

**Did:**
- Created `docs/findex_columns.md`: 23 model features, 1 fairness-audit column (`female`) and 5 identifier/metadata columns, each with a readable name, question text and answer codes
- Checked answer codes against the official World Bank codebook (WLD_2021_FINDEX_v03_M)
- Added a check cell to `notebooks/01_audit.ipynb` confirming all 24 kept columns have data in all 20 countries
- Added `MIN_AGE = 18` and `EXPECTED_APPLICANTS = 18_718` to `config.py`, plus a test (6 tests passing)

**Broke / learned:**
- Many Findex columns are follow-up questions only asked when an earlier answer was "yes", so a high missing share isn't always a data error
- "Don't know", "refused" and "does not apply" codes (e.g. 3/4, or 4/5/6) must become missing values, not real answers
- `fin24a`/`fin24b` looked useful but are missing for up to 48% of respondents in some countries; `fin24` covers the same idea with no gaps

**Decisions:**
- Kept 23 features chosen on four tests: data in all 20 countries, something a loan officer could ask, relevant to repayment risk, and not a follow-up question
- `female` is kept for the fairness audit only, never as a model feature
- **Dropped respondents under 18:** Findex surveys people aged 15+, but minors can't legally take loans. This removes 1,312 respondents aged 15–17 and 30 with no recorded age, so the pool goes from 20,060 respondents to **18,718 applicants** (Kenya: 952; smallest country: Liberia, 874)

## 2026-10-06 — Step 1.3: Findex cleaning module

**Goal:** Turn the raw survey codes into clean, readable data and keep adults only.

**Did:**
- Wrote `src/mkopoguard/data/clean.py`: selects and renames 29 columns, decodes answer codes into values, turns don't know / refused / does not apply into missing, and drops respondents under 18 or with no age
- Added 18 unit tests (`tests/unit/test_clean.py`) using small hand-built rows with known answers
- Added 4 real-data tests (`tests/data/test_findex_real.py`) that skip automatically in CI when the CSV isn't present
- Saved `data/interim/findex_clean.parquet`: 18,718 adult applicants
- Verified the last six column codes against the codebook
- 28 tests passing locally (CI: 24 passed, 4 skipped)

**Broke / learned:**
- `AttributeError: module 'mkopoguard.config' has no attribute 'MIN_AGE'`: the notebook kernel kept an old copy of `config.py` in memory. Fixed by restarting the kernel; added `%load_ext autoreload` / `%autoreload 2` as the first cell to stop it happening again
- `.map(dict)` turns any code not in the dictionary into NaN, which handles don't know / refused automatically
- `Int8` (capital I) is pandas' nullable integer type; plain `int` columns can't hold missing values
- `pytest.mark.parametrize` runs one test function over many inputs
- `pytest.mark.skipif` lets data-dependent tests run locally but skip in CI

**Decisions:**
- Yes/no answers stored as 1/0; categorical answers stored as readable labels (e.g. `cash_only`, `could_not`)
- Kept the 11 respondents aged 99: most likely the survey's top value for "99 or older"
- The cleaning function raises a clear error if an expected column is missing, rather than failing later

## 2026-10-06 — Step 1.4: Simulator design (data card)

**Goal:** Decide on paper how each applicant's loan, mobile money statement and repayment outcome will be generated, before writing any simulator code.

**Did:**
- Wrote `docs/data_card.md`: what's real vs simulated, timeline, hidden traits, loan rules, transaction types, default rule, reproducibility and known limitations
- Checked the mobile money split in the cleaned data: 9,848 applicants (53%) with mobile money, 8,870 (47%) without; Kenya 688 / 264

**Broke / learned:**
- A model can only learn what the simulator puts in. With no noise, or with the default rule built from the same features the model sees, scores look near-perfect but prove nothing
- **Hidden traits** (true income, discipline, stress) drive both transactions and default, but the model only sees their traces in the statement. That's why mobile money data can add information beyond the survey, which the ablation in step 3.7 will test
- A realistic target is ROC-AUC around 0.70–0.85; much higher would mean the simulator is too easy

**Decisions:**
- **Country incomes:** all 20 countries' income quintiles treated as Kenya-equivalent, with no GDP multiplier. Simpler, no extra data source, and the app scores Kenyans. Limitation recorded in the data card
- **No mobile money means no statement:** these 8,870 applicants are thin-file borrowers (`has_statement = 0`), judged on survey answers alone. This matches the original proposal's focus on first-time and informal borrowers. Results will be reported for both groups
- Gender and country are never used in the default rule, to keep the Phase 4 fairness check honest
- Overall default rate target about 20% (band 15–25%); `SIMULATION_SEED = 42`

**Next:** Step 1.5: build the population (load the 18,718 cleaned applicants and draw hidden traits with a fixed seed).

## 2026-10-07 — Step 1.5: Build the population

**Goal:** Turn the 18,718 cleaned respondents into loan applicants with hidden traits, reproducibly.

**Did:**
- Wrote `src/mkopoguard/data/population.py`: applicant IDs (`A-00001`), four hidden traits (monthly income, income pattern, discipline, stress), application dates in 2025, a 180-day statement window, and `has_statement`
- Added simulation settings to `config.py`: seed, dates, income medians by quintile
- Wrote `scripts/build_population.py`, which saves `data/interim/population.parquet` and prints a fingerprint
- Added 18 unit tests and 3 real-data tests (49 passing)

**Broke / learned:**
- One random generator, `np.random.default_rng(seed)`, with draws in a fixed order is what makes output reproducible. New draws must go after existing ones, or every later number changes
- Sorting rows before assigning IDs means the same people in a different order still get identical results, and a test proves it
- Log-normal income gives a realistic shape: most people near the median, a long tail of high earners, nobody negative
- **My fingerprint matched the reference run exactly (4566875540482660526)** on a completely different machine and operating system

**Decisions:**
- Every hidden column starts with `hidden_` so it can be filtered out and tested for later
- Result: 59% irregular income, 27% salaried, 14% seasonal; median income KES 11,130, which fits informal-sector borrowers

## 2026-10-07 — Step 1.6 (part 1): Loans

**Goal:** Give every applicant a loan amount and repayment term (data card section 2.3).

**Did:**
- Wrote `src/mkopoguard/data/loans.py`: amount = income × a log-normal ratio (median 1.0, 0.3–3.0), rounded to KES 500, kept within KES 1,000–100,000; term of 1, 3 or 6 months by amount band
- Started `scripts/generate_synthetic.py`
- Added 10 unit tests and 2 real-data tests (61 passing)

**Broke / learned:**
- Each simulation stage gets its own random stream (`default_rng([seed, stage])`), so changing one stage never reshuffles another
- `np.searchsorted` on cumulative probabilities picks a weighted option for thousands of rows at once
- Statistical tests (e.g. term shares within 2 points of the config over 20,000 draws) catch broken logic without ever failing randomly, because the seed is fixed

**Decisions:**
- Stored `hidden_loan_to_income`, the true ratio after rounding and clipping, as hidden because it uses the true income
- Result: median loan KES 11,000; terms 7,075 × 1 month, 7,455 × 3, 4,188 × 6. Fingerprint matched the reference exactly


## 2026-10-07 — Step 1.6 (part 2): Spending behaviour and statements

**Goal:** Generate six months of mobile money activity for the 9,848 applicants with a statement.

**Did:**
- Wrote `src/mkopoguard/data/transactions.py`: hidden habits for all applicants (overdraft use, repayment readiness, betting share, missed bills), then statements that plan every transaction and apply them in time order with a running balance and overdraft rules
- Generated 1,700,330 transactions in about 34 seconds
- Added 21 unit tests and 4 real-data tests (86 passing)

**Broke / learned:**
- **Found a design bug by checking results against the data card:** overdraft repayment didn't depend on discipline (everyone repaid ~94%) because every payday gave another chance to repay. Fixed by making low-discipline applicants repay less often and never more than half their balance. Now 34% repaid for the least disciplined quarter vs ~90% for the rest
- Checked the other links work: betting is 25% of spending for the least disciplined vs 3% for the most; overdraft borrows 4.0 vs 0.8 between the most and least stressed; statement income matches true income (ratio 1.0)
- Balances can only be computed one transaction at a time, so that part is a plain loop, while planning events uses vectorised NumPy
- `rng.dirichlet` splits a total into random parts that add up exactly
- `test_balances_add_up_row_by_row` checks every balance change equals its signed amount

**Decisions:**
- Every applicant's statement has its own random stream, so one statement never depends on another (proved by a test)
- Random stream numbers: loans 1, behaviour 2, transactions 3, defaults 4
- Transactions fingerprint matched the reference exactly


## 2026-10-07 — Step 1.6 (part 3): Defaults

**Goal:** Decide who defaults using the data card's risk rule, at about a 20% default rate.

**Did:**
- Wrote `src/mkopoguard/data/defaults.py`: 11 weighted drivers plus noise, with the intercept `b0` solved by bisection so the average default probability is exactly 20%
- Recorded the final weights and results in `docs/data_card.md` (section 4)
- Added 17 unit tests and 2 real-data tests (105 passing)

**Broke / learned:**
- `@` (matrix multiplication) applies 11 weights to 18,718 applicants in one line
- Bisection makes the default rate self-correcting: change a weight and `b0` re-adjusts
- **Fingerprint mismatch investigated:** my applicants fingerprint differed from the reference while transactions matched. Narrowed it down with smaller fingerprints: loans and default outcomes were identical (3,723 defaults, same labels hash); only hidden decimal columns differed, from NumPy 2.5.3 vs 2.4.4 computing `exp()` differently in the last decimal place. Pinned versions in `requirements.txt` keep my runs and CI consistent

**Decisions:**
- Results: default rate 19.9% (Kenya 16.1%); poorest quintile 25.1% → richest 16.1%; salaried 13.2%, seasonal 18.5%, irregular 23.3%
- Ceiling ROC-AUC with the true probabilities is 0.84, so the simulator isn't too easy. Quick ablation preview: survey only 0.65, statement only 0.69, both 0.73
- **Fairness finding:** women default at 21.2% vs 18.3% for men even though gender is never used (a test proves flipping gender or country changes nothing). The gap comes through correlated features like income and employment. To audit in Phase 4
- Kept my current fingerprints as my reference


## 2026-10-07 — Step 1.7: Data validation with Pandera

**Goal:** Formally validate both generated files against written rules, and refuse to save invalid data.

**Did:**
- Installed Pandera 0.34 (added to `requirements.txt`)
- Wrote `src/mkopoguard/data/schemas.py`: rules for applicants (IDs, countries, ages, survey codes, loan limits, probabilities, 180-day window, statement only with mobile money) and transactions (types, directions, positive amounts, no negative balances), plus cross-table checks
- The generator now validates before saving (about 3 seconds on the full data)
- Added 16 schema tests and 1 real-data check (122 passing)

**Broke / learned:**
- **Pandera caught a real type mismatch on its first run:** `int` means 64-bit, but 0/1 flags are stored as `int8` and age as `int16`. The schema now states the true types, which is exactly what schemas are for
- pandas 3 stores dates in microseconds and text as a new `str` type, so the schemas check values rather than low-level types
- The CI test fixture runs the **whole Phase 1 pipeline** on 80 made-up respondents: an end-to-end test without the real data
- **Negative testing:** each rule is proven by a test that breaks the data and expects a failure. A schema that accepts everything would still pass the "valid data" test
- `lazy=True` reports every failure at once, not just the first

**Decisions:**
- Allowed survey values come straight from the cleaning mappings, so there's one source of truth
- `full=True` adds whole-dataset checks (18,718 applicants, all 20 countries, default rate in band)


## 2026-10-07 — Step 1.8: Leakage controls (Phase 1 complete)

**Goal:** Make sure nothing the model shouldn't know can reach it.

**Did:**
- Wrote `src/mkopoguard/features/columns.py`: a column guard that blocks hidden traits (any `hidden_` column, even future ones), the outcome, IDs, dates and `is_female`, leaving 27 allowed columns
- Added `docs/data_card.md` section 5 documenting the four leakage layers
- Added 11 leakage tests and 1 real-data check (134 passing)

**Broke / learned:**
- Two kinds of leakage: **time** (anything after the application date) and **column** (inputs the model must never see)
- **Defence in depth:** four independent layers (generator cap, schema, tests, column guard), so one bug can't let a leak through
- **Snapshot test:** pinning the exact 27 allowed columns forces a deliberate decision whenever a new column appears
- **Stress test:** 150 worst-case statements (low income, heavy betting and overdrafts) and none cross the application date

**Decisions:**
- The guard lists what's forbidden and allows everything else
- `country_code` is allowed for now; Phase 3 tests with and without it

**Phase 1 summary:** 18,718 applicants from real Findex data across 20 countries; 1.7 million simulated transactions; 19.9% default rate; reproducible from one command; validated by schemas; documented in the data card; 134 tests passing.

**Next:** Phase 2, step 2.1: EDA on the real Findex data
