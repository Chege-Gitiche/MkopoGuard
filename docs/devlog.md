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
