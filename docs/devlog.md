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

## 2026-10-08 — Step 2.1: EDA on the real Findex data

**Goal:** Describe who the 18,718 applicants are, using only the real survey answers.

**Did:**
- Wrote `src/mkopoguard/stats.py`: `weighted_share` (survey-weighted share per group) and `pool_average` (gives each of the 20 countries equal weight), with 7 unit tests
- Wrote `src/mkopoguard/viz.py`: one colour-blind-safe chart style for the whole project (blue = Kenya / first series, orange = second, grey = other countries), with `ranked_bars` and `paired_bars`
- Built `notebooks/02_eda.ipynb` and saved five charts to `docs/img/` (`eda_01` to `eda_05`), each with a one-line takeaway
- Recorded a new limitation in the data card

**Findings:**
- Mobile money: Kenya leads at 71%; Mozambique lowest at 29%
- Gender gap: women trail men for bank accounts (32% vs 43%) and mobile money (43% vs 51%), so fewer women will have a statement — a fairness risk for Phase 4
- Mobile money doubles with income across the pool (31% → 63%); Kenya 51% → 83%
- 77% of Kenyan adults borrowed last year, but only 21% from a financial institution vs 55% from family or friends — most borrowers have no formal credit history
- 20% of the poorest quintile couldn't raise emergency money in 30 days vs 6% of the richest

**Broke / learned:**
- Survey weights only balance people **within** one country, so shares must be computed per country first and then averaged, never pooled
- **The 40% country rule was unweighted and included 15–17 year olds.** Weighted and adults-only, seven pool countries sit at 37–39% and Mozambique at 29%. Kept the pool (the rule only served to pick mobile-money markets) and documented it honestly in the data card
- Overlapping axis labels on the borrowing chart: fixed by wrapping long labels
- Chart titles should state the finding, not just the topic

**Decisions:**
- Replaced the guide's "borrowing by urban/rural" chart (urban/rural was dropped in the audit) with borrowing sources and lack of emergency funds
- Didn't chart Kenya by income quintile for emergency funds: about 190 respondents per quintile is too few for a reliable share

## 2026-10-08 — Step 2.2: EDA on the synthetic data (simulator fix)

**Goal:** Check that the simulated loans, statements and defaults behave as the data card says.

**Did:**
- Explored the generated data: default rates by country (16–27%) and income quintile (25% → 16%), loan amounts (median KES 11,000), transactions per applicant, busiest days (max 18 a day), balances
- **Found two implausible results** and fixed them in `data/transactions.py`:
  1. Money piled up: the median applicant ended the statement holding **4.3 months of income** (some over KES 1.5 million), because statements only spent about half of income
  2. Overdraft debt had no ceiling: some applicants owed **11.6× monthly income**
- Fix 1: new `cash_out` transaction — after every income or family payment, each applicant withdraws a fixed share (45–75%, drawn once per person) as cash
- Fix 2: overdraft credit limit of one month's income
- Added `cash_out` to the schema's allowed types and 3 regression tests to `test_transactions.py`
- Wrote `src/mkopoguard/data/checks.py`: 12 plausibility limits (default rates, loan size, transactions per applicant and per day, final balances, overdraft use and debt), now enforced every time data is generated; 15 unit tests + 1 real-data check
- Added a `histogram` helper to `viz.py`; built `notebooks/03_synthetic_eda.ipynb` with five charts (`eda_06` to `eda_10`), including a before/after chart of final balances
- Added section 6 to the data card documenting the changes

**Results after the fix:**
- Median final balance 0.56 months of income (99th percentile 3.5); overdraft debt capped at 1.0 month; 23% of statement holders use overdrafts
- Repaid share: 22% for the least disciplined quarter vs 77% for the most — a much clearer signal than before
- 2,121,944 transactions; transactions fingerprint 11234098556618059811
- **Applicants fingerprint unchanged (15016200708709864706) and still 3,723 defaults** — default outcomes depend only on hidden traits, so the fix didn't touch them

**Broke / learned:**
- **Schemas and plausibility checks answer different questions:** schemas ask "is each value allowed?", plausibility asks "does the data as a whole make sense?" Every value in the old data was valid, but the whole was unbelievable
- **Negative test for the checks:** the old step 1.6 data fails exactly the three balance and debt checks and nothing else, proving the checks target the real problem
- Adding one random draw (`cash_share`) shifts every later number in that applicant's stream, which is why the transactions fingerprint changed; separate streams kept applicants and defaults identical
- Loans are rounded to KES 500, so narrow histogram bins showed false gaps at the low end — used wider bins
- The transactions chart has two humps: irregular earners (median 273 rows) get many small payments each followed by a withdrawal, vs salaried (139) and seasonal (103)

**Decisions:**
- Kept a copy of the old statements (`data/interim/transactions_step16.parquet`) for the before/after chart
- Plausibility limits are judgement calls, recorded in the data card

## 2026-10-08 — Steps 2.3 and 2.4: Statement features and known-answer tests

**Goal:** Turn 2.1 million transactions into one row of features per applicant, and prove each feature is calculated correctly.

**Did:**
- Wrote `src/mkopoguard/features/transactions.py`: 17 statement features (all prefixed `stmt_`) covering income, spending, overdrafts, balances, activity and loan size relative to statement income
- Wrote `scripts/build_features.py`: builds `data/processed/features.parquet` — 18,718 rows × 47 columns (allowed model inputs + statement features + `defaulted` + `is_female` for the fairness audit). Hidden traits never enter the file
- 14 known-answer unit tests (`tests/unit/test_features.py`) and 4 real-data checks
- Built `notebooks/04_features.ipynb` with a signal-strength chart (`features_01_signal_strength.png`)
- Extended `ranked_bars` to show non-percentage values and start bars at a baseline (AUC's "no signal" point is 0.5)

**Findings:**
- Strongest single feature: betting share (ROC-AUC 0.653), then overdraft behaviour (0.58–0.59)
- Money from family/friends has no signal (0.502), as designed — remittances aren't in the default rule
- "Days since last income" looks backwards (recent income = riskier) but is right: irregular earners are paid every few days, and irregular income is riskier
- All 17 together in a simple logistic regression: ROC-AUC 0.71 (statement holders only)
- Overlap: average and median inflow 0.92 correlated; overdraft features 0.74–0.82

**Broke / learned:**
- **A test found a real edge case:** a statement with income but no spending got missing spending shares (division by zero outflow). Never happens in the generated data, but an uploaded statement in the API could. Fixed: no spending means a 0% share, locked in by a regression test
- **Training–serving skew:** the same feature function runs in training and in the API, so features can't be computed differently in the two places
- Empty months must count as zero income, or someone paid once in six months looks perfectly steady
- Volatility is the coefficient of variation (std ÷ mean), which judges small and large earners on the same scale
- Known-answer tests: e.g. 18,000 in one month and nothing in five gives volatility √5 ≈ 2.24, worked out on paper first
- The feature function drops transactions on/after the application date or older than 180 days itself — a fifth leakage layer, tested with planted KES 99,999 bets

**Decisions:**
- Thin-file applicants get missing values, not zeros: "no statement" is different from "spent nothing on betting"
- Dropped the guide's "distinct income sources" feature: statements don't record who sent money (guide updated)
- Step 2.4 is covered by the known-answer tests written here

**Next:** Step 2.5: preprocessing pipeline.

## 2026-10-08 — Step 2.5: Preprocessing pipeline

**Goal:** Turn the model table into clean numbers inside one scikit-learn pipeline that can't leak.

**Did:**
- Wrote `src/mkopoguard/features/preprocess.py`: three lanes in a `ColumnTransformer`
  - Numbers (21): log money amounts → clip to 1st/99th percentile → median fill → scale
  - Yes/no (16): fill with the most common answer
  - Categories (6): missing answers become a `missing` category → one-hot
- Wrote a custom `Winsorizer` transformer (scikit-learn has no "clip to learned percentiles" step)
- `make_pipeline(model)` wraps preprocessing and model into one object to fit, cross-validate and save together
- 13 unit tests + 1 real-data check (192 passing)
- Added a pipeline check to `notebooks/04_features.ipynb`

**Results:**
- 43 input columns → 68 clean numeric columns, no missing or infinite values, about 1 second to fit
- Quick logistic regression (5-fold CV, all applicants): ROC-AUC **0.720** without country, 0.719 with it

**Broke / learned:**
- **Everything learned (medians, clip bounds, means, category lists) is learned in `fit()` only.** Inside a Pipeline, cross-validation re-learns it per fold and the test set never contributes — a test proves transforming new data never changes what was learned
- Logging money first stops a few huge earners squashing everyone else into a tiny range after scaling
- `handle_unknown="ignore"` means an unseen category at prediction time doesn't crash the API
- `remainder="drop"` silently excludes `applicant_id`, `defaulted` and `is_female` — one more barrier against leakage
- Pandas nullable integers (`Int8` with `<NA>`) need converting to floats before scikit-learn's imputers
- `tmp_path` gives each pytest test its own temporary folder

**Decisions:**
- Thin-file statement features are filled with the training median; `has_statement` tells the model they're filled in
- `country_code` is off by default (adds nothing); Phase 3 confirms properly
- Three tests protect the future API: one applicant alone matches the same applicant in a batch, a saved/reloaded pipeline gives identical output, and unknown categories don't crash

**Next:** Step 2.6: train/validation/test split (70/15/15, stratified by country and default).
