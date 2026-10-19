# Test strategy

**Product:** MkopoGuard, a microloan default-risk scorer for loan officers (FastAPI service, a
scikit-learn model with reason codes, and a plain HTML/CSS/JavaScript frontend).
**Owner:** Emmanuel Chege Gitiche (developer and tester). **Version:** 1.0, 2026-10-19.

## 1. Scope

In scope: everything that ships in the repository.

- **Data:** the synthetic population, M-Pesa-style transaction logs, cleaning, the train /
  validation / test split, and leakage rules.
- **Features and model:** feature engineering, preprocessing, the champion model bundle, the
  cost-based decision policy, calibration, reason codes, behaviour and fairness checks.
- **API:** login, sessions and roles, `/predict` with statement upload, history, model info,
  rate limiting, security headers, error handling, database migrations.
- **Frontend:** the six screens, the assessment flow, accessibility, the disclaimer.
- **Deployment:** the live site's smoke test (Phase 8).

## 2. Approach

Risk-based: the most effort goes where a failure would hurt an applicant or leak data.

| Risk | How bad | Where it is tested |
|---|---|---|
| A wrong score or decision reaches an officer | Critical | Model tests (quality floor, golden examples, behaviour), API tests that replay the samples |
| One officer sees another officer's assessments | Critical | API and security tests (404 for anyone else's record, owner filter inside the query) |
| Login bypassed or a token forged | Critical | Security tests (tampered, expired and wrong-audience tokens, cookie guard) |
| Personal data in logs or responses | High | Unit tests on the log filter and on 422 responses that never echo input |
| A statement parsed wrongly | High | Unit and data tests on the parser, balance chain and coverage rules |
| The model treats groups unequally | High | Fairness audit; known gaps are documented and accepted for v1 (Section 7) |
| A screen unusable by keyboard or screen reader | Medium | axe, Lighthouse and keyboard tours |
| Slow scoring under load | Medium | Locust load test of `/predict` |

Every requirement gets an ID and at least one test (traceability matrix in
`docs/test_plan.md`, step 7.2). Every fixed bug gets a regression test (`docs/bug_log.md`).

## 3. Test layers

| Layer | What it checks | Tool | Folder | Runs |
|---|---|---|---|---|
| Data | Schemas, ranges, leakage (no transaction after its loan date) | pytest, project data checks | `tests/data` | Every push |
| Unit | Cleaning, features, statement parser, reason codes, helpers | pytest, Hypothesis | `tests/unit` | Every push |
| Model | Quality floor, calibration, determinism, behaviour, golden examples | pytest | `tests/model` | Every push |
| API | Status codes, validation, auth, roles, data isolation, migrations | pytest + httpx | `tests/api` | Every push |
| Security | Tampered tokens, CSRF guard, rate limit, injection, headers | pytest | `tests/security` | Every push |
| End-to-end | Real browser journeys: login, assess, result, history, logout | Playwright | `tests/e2e` | Pull requests to `main` |
| Accessibility | No serious axe violations, Lighthouse 90 or more, keyboard use | Playwright + axe-core, Lighthouse | `tests/e2e` | Pull requests; Lighthouse by hand per release |
| Performance | `/predict` latency at 10 users | Locust | `perf/` | Before each release |

Today the automated tests live in `tests/unit` and `tests/data`; step 7.4 moves the model, API
and security tests into their own folders and adds the end-to-end and load tests.

## 4. Environments

| Environment | Used for | Details |
|---|---|---|
| Local | Development, all layers | Windows 11, Git Bash, Python virtual environment, SQLite, Chrome |
| CI | Every push and pull request | GitHub Actions on Ubuntu, fresh SQLite database, headless Chromium (step 7.5) |
| Live | Smoke and load tests after deployment | Free-tier host and Supabase Postgres (Phase 8) |

Test data is synthetic only: the generated population and the ten committed sample applicants.
Tests never use the developer's own database or `.env` file.

## 5. Entry and exit criteria

**Entry** (before a change is tested): it runs locally, `ruff` passes, and new behaviour has its
own tests. End-to-end and load tests also need the app running with a migrated database and the
packaged model bundle.

**Exit** (before a release is tagged or deployed):

- All automated tests pass. The only expected failures are the three known behaviour flips,
  kept as strict expected failures so a fix is noticed.
- Line coverage of `src/mkopoguard` is 80% or more.
- Model: test PR-AUC 0.35 or more (a random guess scores the default rate, about 0.20);
  calibration error (ECE) 0.03 or less; the golden examples replay exactly.
- Every requirement in the traceability matrix has at least one passing test.
- No open critical or major (S1, S2) defects.
- Accessibility: Lighthouse 90 or more on every screen; no serious or critical axe violations.
- Performance: `/predict` p95 under 500 ms locally and under 1.5 s on the live free tier, at
  10 concurrent users, with no errors.
- The production smoke test passes (Phase 8).

## 6. Out of scope

- Real applicants, real M-Pesa or credit-bureau data, and model accuracy on any real population.
- Legal or regulatory sign-off (Data Protection Act 2019, CBK lending rules). Only the design
  choices are documented.
- Browsers other than Chromium-based ones in automated tests (Chrome is also checked by hand).
- Load above 10 concurrent users, soak tests, and professional penetration testing.
- Languages other than English.

## 7. Known risks accepted for v1

- **Fairness:** the audit found women and applicants aged 50 and over are declined more often
  than others with the same true risk. Documented in the model card; addressed in Phase 9.
- **Behaviour:** for three inputs the model moves the score the wrong way (for example, more
  overdraft borrowing lowers it slightly). Kept as strict expected failures; addressed in Phase 9.
- **Logout:** login tokens are not stored on the server, so a copied token works until it
  expires (at most 30 minutes).

## 8. Defects

Logged in `docs/bug_log.md` with ID, date, summary, severity, steps, root cause, fix commit and
regression test.

| Severity | Meaning |
|---|---|
| S1 Critical | Wrong score or decision, another officer's data exposed, login bypass, data loss |
| S2 Major | A main task can't be done and there is no workaround |
| S3 Minor | A workaround exists, or a message or layout is wrong |
| S4 Cosmetic | Wording, spacing, appearance |

## 9. Deliverables

This strategy; the test plan with requirements, traceability and test cases (`docs/test_plan.md`);
the automated suites; CI with a coverage report; the bug log; exploratory testing notes; and the
test summary report (`docs/test_summary.md`).
