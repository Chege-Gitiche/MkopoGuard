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

## 2026-10-9 — Step 2.6: The one-and-only split

**Goal:** Split the applicants into train, validation and test sets once and lock them, so every model in Phase 3 is judged on the same applicants and the test set stays unseen until the end.

**Did:**
- Wrote `data/split.py`: a 70/15/15 split stratified by country × defaulted × has_statement (80 groups). Every part has the same country mix, default rate and thin-file share as the whole pool.
- Result: 13,102 train / 2,808 validation / 2,808 test.
- Only applicant IDs are saved, in `data/splits/applicant_split.csv` (committed to Git), so rebuilding the features never changes who is in which set.
- `write_once()` refuses to overwrite a split that differs from the saved one. `--force` exists only for when the applicants themselves change.
- `load_split()` locks the test set: it raises an error unless you pass `allow_test=True`, which happens exactly once, at step 3.12.
- Tests: the split is reproducible and doesn't depend on row order, the parts don't overlap and cover everyone, proportions hold within each group, and the test set can't be loaded by accident.

**Broke / learned:**
- The split has to depend only on the applicants and the seed, never on row order. That's why the table is sorted by `applicant_id` before splitting.
- A test set protected by good intentions isn't protected. Making the code refuse is the QA way to do it.

**Decisions:**
- Stratify on three things, not just the default flag. Kenya and thin-file applicants are the groups we report on, so every part must have enough of them.
- Commit the split file to Git so it's part of the project's history, not something that gets regenerated.

## 2026-10-9 — Step 3.1: Experiment tracking with MLflow

**Goal:** Record every model run automatically, including the things that are easy to forget, so any result can be traced back to the exact code and data that produced it.

**Did:**
- Wrote `tracking.py`. `start_run()` tags every run with the Git commit (with `-dirty` if there were uncommitted changes), fingerprints of the split file and the model table, the step number, and which split the metrics were measured on (cv / train / validation / test).
- Local store: SQLite at `mlruns/mlflow.db`, git-ignored. View with `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- Turned MLflow's usage telemetry off before importing it. A credit-risk project shouldn't send anything out unasked.
- Models are saved in skops, a safe format. Loading only accepts types on an explicit allow-list (`SKOPS_TRUSTED_TYPES`), so a tampered model file can't run code.
- Added `scripts/tracking_smoke.py`: it trains a dummy model, logs it, reloads it, and checks the predictions are identical.

**Broke / learned:**
- The first reload failed with "untrusted types". Skops rejected our own `Winsorizer` and some numpy and sklearn types. Fixed by listing them in `SKOPS_TRUSTED_TYPES`. Every new model type has to be added there (tree models in 3.4, imblearn in 3.5).
- On Windows, the tracking URI has to be built with `as_posix()`. Backslashes break the `sqlite:///` path.

**Decisions:**
- Local SQLite, not a hosted server: free, works offline, and enough for one person.
- `evaluated_on` is a required tag, so a test-set score can never be mistaken for a validation score

## 2026-10-9 — Step 3.2: The evaluation plan, decided before any model

**Goal:** Fix the metrics and rules first, so model choice can't be bent to fit whatever result looks good.

**Did:**
- Wrote `models/metrics.py`. `evaluate()` returns the same numbers for every model:
  - ranking: **PR-AUC (primary)**, ROC-AUC, KS
  - probabilities: Brier, Brier skill, ECE
  - decisions at a provisional threshold of 0.20: recall, precision, false-positive rate, accuracy, approval rate
- Added bootstrap confidence intervals (1,000 resamples), reporting by subgroup (all / Kenya / statement holders / thin-file), and a calibration table.
- Added a **leakage alarm**: stop if ROC-AUC > 0.86 or PR-AUC > 0.65. Those ceilings sit above what the simulator's own hidden rule could plausibly achieve, so a score above them means information is leaking.
- Wrote `docs/evaluation_plan.md` with the metrics and rules 1–5.

**Broke / learned:**
- With 20% defaulters, accuracy is misleading: approving everyone scores 80%. PR-AUC focuses on how well the model finds defaulters, which is the job.
- A good probability and a good ranking are different properties. ECE checks the first, PR-AUC the second.

**Decisions:**
- PR-AUC is the deciding score. Calibration (ECE) and approval rate are guardrails.
- The 0.20 threshold is provisional. The real one comes from the cost of each kind of mistake in step 3.11.

## 2026-10-9 — Step 3.3: Experiment harness and baselines

**Goal:** Put every model through one identical procedure, and set the bar every later model has to beat.

**Did:**
- Wrote `models/experiment.py`. `run_experiment()` runs 5-fold CV on train (folds stratified like the split), fits on all of train, records the training score to measure overfitting, scores validation overall and per subgroup, checks the leakage alarm, and logs everything to MLflow (CSV tables plus the pipeline).
- `scripts/train_baselines.py` results (CV PR-AUC):

  | model | CV PR-AUC | val ECE | val approval |
  |---|---|---|---|
  | dummy (no skill) | 0.199 | 0.000 | 100% |
  | logistic | **0.393** | 0.011 | 60% |
  | logistic + class weights | 0.392 | 0.252 | 10% |
  | logistic + country | 0.394 | 0.010 | 60% |

**Broke / learned:**
- Class weights didn't improve ranking, but they broke the probabilities: ECE went up about 20 times and the approval rate fell to 10%.
- Adding country gained 0.001, far smaller than the fold-to-fold noise (std 0.029). That led to a new rule.

**Decisions:**
- **Rule 6 (tie rule):** a gain smaller than one fold standard deviation is a tie, and a tie goes to the simpler model.
- `country_code` stays out of the model. It adds nothing and invites unfairness across countries.
- Baseline to beat: plain logistic regression, CV PR-AUC 0.393.

## 2026-10-9 — Step 3.4: Tree-based candidates

**Goal:** See whether more flexible models (random forest, XGBoost, LightGBM) beat logistic regression.

**Did:**
- Wrote `models/candidates.py` with conservative starting settings (shallow trees, a minimum number of applicants per leaf) and `scripts/train_candidates.py`.
- Results (CV PR-AUC, with training vs validation PR-AUC):

  | model | CV PR-AUC | train | validation |
  |---|---|---|---|
  | logistic (baseline) | **0.393** | 0.41 | 0.41 |
  | XGBoost | 0.381 | 0.57 | 0.40 |
  | random forest | 0.378 | 0.54 | 0.38 |
  | LightGBM | 0.378 | 0.63 | 0.40 |

- Chart: `docs/img/models_01_overfitting.png`.
- Tests: every candidate trains, is reproducible, and saves and reloads with identical predictions.

**Broke / learned:**
- All three tree models overfit and lose to logistic regression. Diagnosis: the simulator's default rule is **additive** (no interactions), which is exactly the shape logistic regression fits. Even on the hidden drivers, logistic beats XGBoost (0.580 vs 0.566). This is now listed as a limitation in the data card.
- Random forest predictions changed in the 16th decimal place from one call to the next when `n_jobs=-1`, because parallel trees add up their votes in a different order each time. `n_jobs=1` fixed it.
- Swapped `xgboost` for `xgboost-cpu`, which avoids about 300 MB of GPU libraries we don't need.
- The `viz.py` patch failed on my machine because `histogram` is at the end of my file. Lesson: write patches against the real file layout and make them order-independent.

**Decisions:**
- Carry logistic and XGBoost forward (the best of the trees) into tuning. Drop random forest and LightGBM.

## 2026-10-9 — Step 3.5: Imbalance strategies

**Goal:** Test the usual advice for imbalanced data (class weights, SMOTE) before adopting it.

**Did:**
- Added an optional `sampler` step to `make_pipeline()` using imbalanced-learn's Pipeline, which resamples **only while fitting**. Predictions and held-out folds always use real applicants.
- Added the imblearn types to `SKOPS_TRUSTED_TYPES`. `run_experiment()` now logs a `sampler` param.
- `scripts/compare_imbalance.py`: none / weights / SMOTE for logistic and XGBoost.

  | run | CV PR-AUC | gain vs none | val ECE | val approval |
  |---|---|---|---|---|
  | logistic_none | 0.393 | – | 0.011 | 60% |
  | logistic_weights | 0.392 | −0.000 | 0.252 | 10% |
  | logistic_smote | 0.391 | −0.002 | 0.241 | 14% |
  | xgboost_none | 0.381 | – | 0.013 | 60% |
  | xgboost_weights | 0.378 | −0.002 | 0.227 | 15% |
  | xgboost_smote | 0.375 | −0.006 | 0.034 | 55% |

- Tests (`test_imbalance.py`, 8 tests): a `RecordingSMOTE` records how many rows it receives. This proves SMOTE only ever sees training rows, never a held-out fold or the applicants being scored. The tests also check that runs are reproducible and that the SMOTE pipeline saves and reloads identically. Total: 279 passed.

**Broke / learned:**
- Neither strategy improves ranking. Weighting raises every score by about the same amount, so the order barely changes but the probabilities stop being honest. SMOTE blends defaulters together, and on additive data that only adds noise.
- If SMOTE runs before cross-validation, invented defaulters built from held-out applicants leak into training and inflate the scores. Running it inside the pipeline prevents that.
- Ruff B007: a loop variable that's never used must start with `_` (`_family`).

**Decisions:**
- Train on the data as it is. The imbalance is handled where it belongs: calibration (3.10) and a threshold based on what each mistake costs (3.11).

**Next:** 3.6, tune logistic and XGBoost with Optuna.

## 2026-10-12 — Step 3.6: Tuning with Optuna

**Goal:** Find out whether better settings let logistic regression or XGBoost beat the untuned champion.

**Did:**
- Wrote `models/tuning.py` and `scripts/tune_models.py`: Optuna (TPE sampler, seed 2026) ran 40 logistic and 60 XGBoost trials. Each trial was scored by 5-fold CV on the training set only.
- Trial 0 is always the untuned model, so the study measures its own gain on the same folds.
- Logistic searched C and the L1/L2 mix (saga solver). XGBoost searched depth (down to 1), learning rate, trees, child weight, subsampling and regularisation.
- The chosen settings are saved to `docs/results/tuned_params.json`. Later steps build models with `tuning.chosen_model()`.
- Results:

  | model | untuned CV PR-AUC | best tuned | gain | 1 fold std |
  |---|---|---|---|---|
  | logistic | 0.393 | 0.398 | +0.005 | 0.029 |
  | XGBoost | 0.381 | 0.397 | +0.017 | 0.023 |

- Tests (14): every suggested setting trains, the defaults are inside the search space, the same seed gives the same study, and the tie rule works both ways.

**Broke / learned:**
- XGBoost's best settings used `max_depth = 1`: one feature per tree, which is an additive model. Given the choice, boosting turned itself into a version of logistic regression, and its overfitting gap fell from 0.168 to 0.002. That confirms the 3.4 diagnosis: the data has no interactions to find.
- Logistic's best settings were almost pure lasso, and lasso switched off 38 of 68 columns at no cost. That fed straight into 3.7.
- Bug caught before running: the XGBoost defaults didn't list `reg_lambda` and `gamma`, so Optuna would have filled them in at random in trial 0. Fixed by adding XGBoost's own defaults.
- The saga solver shuffles rows, so it needs `random_state` to be reproducible.

**Decisions:**
- Neither gain beats one fold std (rule 6), so the untuned settings stay. Picking the best of many trials is also slightly optimistic.
- Champion: plain logistic regression (L2, C = 1).

## 2026-10-12 — Step 3.7: Feature ablation

**Goal:** Find which groups of inputs the model actually needs, so the application form can be shorter.

**Did:**
- Grouped the 43 inputs into 8 blocks the way an applicant experiences them (`features/groups.py`): loan, profile, financial access, income channels, savings & resilience, borrowing history, transfers, statement.
- Added a `drop=` option to `input_columns`, `build_preprocessor`, `make_pipeline` and `run_experiment`. Dropped columns don't even need to be present when predicting, so the API never has to ask for them.
- Removed one group at a time and compared each CV fold with the same fold of the full model (a paired comparison).
- **Rule 7:** a group is needed if removing it causes a consistent loss (bigger than its standard error) that is also meaningful (more than 0.002 PR-AUC), for all applicants OR for thin-file applicants.
- The combined drop had to pass the same test on CV, and then not lose clearly for any subgroup on validation (paired bootstrap).
- Results: savings & resilience (+0.047) and the statement (+0.055) carry the model. Financial access, income channels and transfers are not needed.
- Tests (20), including a known-answer pair: removing the only real signal shows as needed, and removing columns with no information shows as not needed.

**Broke / learned:**
- I nearly dropped borrowing history. An earlier procedure removed groups one after another and dropped it at the end, but the validation check showed thin-file applicants losing clearly (interval 0.005 to 0.036). That's why validation is now a gate on every drop.
- Thin-file applicants have to be checked separately. Loan terms look useless overall (+0.001) but matter for them (+0.003), because without a statement there's no `stmt_loan_to_inflow`.
- A "pure noise" test was flaky across random seeds. I replaced it with columns that carry no information at all (everyone gives the same answer), which gives an exact known answer.
- Known-answer check against the simulator: both groups with no true driver were dropped, and every group with a strong driver was kept. Income channels feeds a weak driver but was dropped, because the statement already carries that signal.

**Decisions:**
- Drop financial access, income channels and transfers (`docs/results/feature_selection.json`).
- Model inputs go from 43 to 32. **Survey questions go from 23 to 12.** CV PR-AUC went up slightly, from 0.3926 to 0.3950.

## 2026-10-12 — Step 3.8: Learning curve

**Goal:** Find out whether more applicants would improve the model, or whether it has learned all the inputs can tell it.

**Did:**
- Trained on 1/16, 1/8, 1/4, 1/2 and all of each CV fold's training rows (3 random draws for each smaller share). Every version was scored on the same full held-out fold.
- Added a ceiling line: the PR-AUC of the simulator's *true* default probabilities (0.617). No model can beat it with any amount of data. It's read through `load_split`, so the test set stays locked.
- Results: logistic gained +0.010 and then +0.003 from the last two doublings. The next doubling is projected at +0.001. XGBoost is still climbing (+0.011 per doubling), with an overfitting gap of 0.21.
- Tests (14): stratified, reproducible subsampling, sizes that really double, a shrinking overfitting gap, the projection never predicting faster gains, and the ceiling beating a trained model.

**Broke / learned:**
- Each CV fold trains on 80% of the training set (10,482 applicants, not 13,102).
- The first ceiling test failed: true probabilities that take only two values give ties, and PR-AUC handles ties badly. Fixed by making the made-up risk continuous.
- The 0.22 gap to the ceiling is information the inputs don't carry (true discipline, income, noise), not a shortage of applicants.
- The leakage alarm (0.65) sits just above the ceiling (0.617), which confirms it was set sensibly in 3.2.

**Decisions:**
- The 18,718-applicant pool is big enough, so I won't grow the simulation. XGBoost would need about 4× the data just to reach logistic's current score.

## 2026-10-12 — Step 3.9: Leave-one-country-out

**Goal:** Check that the model works in a country it has never seen, for a lender entering a new market.

**Did:**
- Wrote `models/country_transfer.py`. In each CV fold, each country's held-out applicants are scored twice: by the fold's model ("seen"), and by the same model retrained without that country ("never seen").
- **Rule 9:** no country loses clearly (above 0.02 PR-AUC, beyond chance, with the interval widened for 20 countries), and no never-seen country is miscalibrated beyond chance (Bonferroni).
- Added an optional `level` to `paired_bootstrap` (the default stays 95%).
- Results: logistic's median loss is −0.0005 and its worst is +0.004 (Zimbabwe). Kenya: 0.433 seen → 0.435 never seen. XGBoost: Kenya −0.025, and Zimbabwe narrowly miscalibrated (z = 3.05 against a bar of 3.02).
- Tests (8) on six made-up countries: "same rule" must transfer, "hidden extra risk in Kenya" must be flagged for calibration, and "a rule only Kenya has" must lose clearly. That last one proves the never-seen model really never saw the country.

**Broke / learned:**
- My first design (train on 19 countries, compare with ordinary CV) made every country look *better* when unseen. The 19-country model had about 12,400 training rows against 10,500 in a CV fold, so it was measuring more data, not transfer. The paired design changes only one thing.
- The same trap appeared in the tests: with 3 small countries, hiding one removed a third of the data and caused false failures. Six countries of 2,000 applicants fixed it.
- **The script failed on my machine with `ModuleNotFoundError: mkopoguard.models.calibration`.** 3.9 reuses the per-country calibration check from 3.10, and I had done 3.9 first. I did 3.10, then re-ran 3.9. Lesson: the guide's order is a dependency order, and a step can quietly rely on a later one.
- Zimbabwe is a watch item: its default rate (27.4%) is above what its applicants' answers predict (22.8%), in both 3.9 and 3.10. The simulator has no country effect, so this is most likely chance.
- Limitation: in the real Findex, `income_quintile` is relative within each country, but the simulator gives each quintile the same income everywhere, so this transfer risk can't show up here.

**Decisions:**
- The champion transfers to new countries. Country stays out of the model. Recheck with real repayment data and monitor calibration.

## 2026-10-12 — Step 3.10: Calibration

**Goal:** Make sure "20% chance of default" really means about 20 in 100, because the threshold in 3.11 depends on it.

**Did:**
- Wrote `models/calibration.py`. It produces out-of-fold predictions for every training applicant and measures ECE, Brier score, calibration slope and intercept, and calibration-in-the-large per group (with the flag bar raised for multiple comparisons).
- Compared no calibration, sigmoid (Platt) and isotonic. Each calibrator learns from an inner 5-fold CV inside the training rows.
- **Rule 8:** adopt a method only if it cuts ECE consistently and by more than 0.005, without worsening the Brier score or costing more than 0.002 PR-AUC.
- Added a **positive control**: the class-weighted logistic from 3.5, which is miscalibrated on purpose.
- Results: champion ECE 0.006, slope 0.96, average predicted 19.9% against an actual 19.9%. The control went from 45% predicted against 20% actual (ECE 0.253) to 19.9% (ECE 0.006) with Platt scaling. No country is flagged; Kenya is predicted at 17.4% against an actual 16.2%.
- Added the calibration types to `SKOPS_TRUSTED_TYPES`, so a calibrated pipeline can be saved to MLflow.
- Tests (21): true probabilities give slope ≈ 1, over-confident ones give slope ≈ 0.5, Platt fixes the control, a deliberately biased country is flagged, and a calibrated pipeline saves and reloads identically.

**Broke / learned:**
- A check that never finds anything proves nothing. The positive control shows the calibration code works when it's needed.
- Multiple comparisons: at the usual 2-standard-error bar, 20 countries give about one false alarm by chance. Zimbabwe (z = 2.5) would have been flagged. Bonferroni raises the bar to 3.02.
- Reliability-chart bins with only a few applicants make wild lines, so bins under 50 applicants are hidden on the chart (they stay in the CSV).
- Logistic regression fitted by maximum likelihood is calibrated by construction. XGBoost is slightly over-confident (slope 0.85).

**Decisions:**
- No calibration (`docs/results/calibration.json`). The probabilities are used as the model gives them.

## 2026-10-12 — Step 3.11: Cost-based decision threshold

**Goal:** Replace the provisional 0.20 cut-off with a rule based on what each decision costs the lender.

**Did:**
- Researched Kenyan digital-lending figures. Fee: 7.5% a month (M-Shwari's 30-day price; Tala, Branch, Okolea and Zenka charge more). Funding cost: 1% a month (assumption; CBK rate 8.75%). Loss given default: 80% (assumption; CBK reports 83.1% of loans under KSh 1,000 default). Sources are in `evaluation_plan.md`, and all three are settings in `config.py`.
- Derived the rule: approve when p < m·T / (m·T + LGD). The loan amount cancels out, but the term doesn't.
- Thresholds: 0.075 (1-month loans), 0.196 (3-month), 0.328 (6-month).
- Wrote `models/decision.py`, with `decide(p, term, policy)` for the API, and saved the policy to `docs/results/decision_policy.json`.
- Results (CV profit per 1,000 applicants): approve everyone KSh 0.85M, provisional 0.20 KSh 1.40M, best single cut-off KSh 1.42M, **cost-based rule KSh 1.67M (+19%)**.
- Validation: +KSh 0.87M against approving everyone (significant). +0.06M against the provisional 0.20, which isn't significant on 2,808 applicants.
- A sensitivity grid (16 combinations of margin and LGD): the rule is profitable in all of them, while approving everyone loses money in 6.
- Tests (19): the threshold is exactly the break-even point, and with known true risk, expected profit matches actual, the rule beats every single cut-off, and profit peaks at scale 1.0.

**Broke / learned:**
- The thresholds really are optimal. Scaling all three up or down, profit peaks at 1.05×, and expected profit matched actual within 2%. That's 3.10's calibration paying off.
- Only 15% of 1-month loans get approved: a short loan earns 6.5% if repaid but loses 80% on default. That matches CBK's finding on tiny, short loans. The fix would be pricing or loan term, not a looser rule.
- Thin-file applicants are approved 34% of the time, against 62% for statement holders. Uploading a statement roughly doubles an applicant's chance, which is a design point for Phase 5.
- Limitations: the simulator's default rate doesn't depend on loan term, and the profit model is simplified (flat fee, no partial repayment, no operating costs).

**Decisions:**
- Use the cost-based rule with thresholds per loan term. The API will call `decision.decide()`.
