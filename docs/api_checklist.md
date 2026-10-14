# API checklist (step 5.8)

Every endpoint, run once by hand from the interactive docs at `/docs`. The automated twin is
`scripts/api_smoke.py`; this checklist also covers what a person sees (the docs, the forms).

| | |
|---|---|
| Tester | |
| Date | |
| Server | `http://127.0.0.1:8000` / live URL |
| Model version (from `GET /health`) | |
| Result | 22 of 22 passed |

Before starting: the server is running, you have an officer account, and **Authorize** is
not yet clicked.

## Public

| ID | Endpoint | Steps | Expected | Pass |
|---|---|---|---|---|
| API-01 | page | Open `/docs` | Title "MkopoGuard API", the "not for real lending decisions" note, 5 groups | ✔ |
| API-02 | `GET /health` | Try it out → Execute | 200; `status: ok`, `database: ok`, the model version | ✔ |
| API-03 | any | Look at the response headers of API-02 | `x-request-id`, `x-content-type-options: nosniff`, `x-frame-options: DENY` | ✔ |

## Logging in

| ID | Endpoint | Steps | Expected | Pass |
|---|---|---|---|---|
| API-04 | `GET /auth/me` | Execute without logging in | 401, "Not logged in, or the session has expired" | ✔ |
| API-05 | `POST /auth/login` | Your email + a wrong password → Execute | 401, "Incorrect email or password" | ✔ |
| API-06 | `POST /auth/login` | An email with no account + any password | 401, the **same** message as API-05 | ✔ |
| API-07 | Authorize | Authorize → email in *username*, your password → Authorize | "Authorized"; the padlocks close | ✔ |
| API-08 | `GET /auth/me` | Execute | 200; your email, name and role; no password or hash anywhere | ✔ |

## Scoring

| ID | Endpoint | Steps | Expected | Pass |
|---|---|---|---|---|
| API-09 | `GET /model/info` | Execute | 200; version, test PR-AUC, thresholds for 1/3/6 months, 4 bands | ✔ |
| API-10 | `POST /predict` | Leave the pre-filled profile (SAMPLE-10), no file → Execute | 200; 19.2%, medium, decline, threshold 0.075, `has_statement: false`, 3 + 2 reasons | ✔ |
| API-11 | `POST /predict` | Paste `samples/profiles/sample_07_very_high.json`, choose `samples/statements/sample_07_very_high.csv` | 200; 49.1%, very_high, decline; first reason mentions betting | ✔ |
| API-12 | `POST /predict` | Change `"age"` to `15` in the profile | 422; `loc: ["profile", "age"]` | ✔ |
| API-13 | `POST /predict` | Replace the profile with `{"age": 30}` | 422; every missing field listed | ✔ |
| API-14 | `POST /predict` | Profile of sample_01, statement of sample_07 (wrong period for it) | 422; `loc: ["statement", "period"]` | ✔ |
| API-15 | `POST /predict` | Choose a non-CSV file (e.g. a photo) as the statement | 422; `loc: ["statement", "file"]`; no stack trace | ✔ |

## History

| ID | Endpoint | Steps | Expected | Pass |
|---|---|---|---|---|
| API-16 | `GET /predictions` | Execute | 200; API-11 first, then API-10; `total` counts them | ✔ |
| API-17 | `GET /predictions` | `band = very_high` | Only very_high predictions | ✔ |
| API-18 | `GET /predictions` | `size = 500` | 422 (the maximum is 100) | ✔ |
| API-19 | `GET /predictions/{id}` | The `id` from API-11 | 200; same probability, threshold, reasons, and the profile answers | ✔ |
| API-20 | `GET /predictions/{id}` | `id = 999999` | 404, "Prediction not found" | ✔ |
| API-21 | `GET /predictions/{id}` | An id made by **another** officer (create a second account to try) | 404, exactly the same as API-20 | ✔ |

## Logging out

| ID | Endpoint | Steps | Expected | Pass |
|---|---|---|---|---|
| API-22 | Authorize | Authorize → Logout, then run `GET /predictions` | 401 | ✔ |

Notes and failures (with the `x-request-id` of the response):
