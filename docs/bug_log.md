# Bug log

Every defect found while building and testing MkopoGuard: what it was, how bad (the scale in
`docs/test_strategy.md`, section 8), how to make it happen, why it happened, the commit that
fixed it, and the test that keeps it fixed. Every fixed bug has a regression test; an open bug's
test is a strict expected failure, so the day it is fixed the test fails and says so.

`tests/unit/test_bug_log.py` checks this file on every run: IDs in order, a real date and
severity, a commit hash or "Open", and regression tests that exist.

BUG-001 and BUG-002 were the first bugs logged here (step 7.4). BUG-003 to BUG-008 were found
in Phase 6, before this log existed, and were recorded afterwards from the devlog; their date is
the day the fix was committed.

## Bugs

| ID | Date found | Summary | Severity | Steps to reproduce | Root cause | Fix commit | Regression test |
|---|---|---|---|---|---|---|---|
| BUG-001 | 2026-10-19 | A statement with a bare carriage return made `/predict` answer 500 instead of reading or refusing it | S3 Minor | Re-save a sample statement in Excel for Mac as "CSV (Macintosh)" (lines end in `\r`), or upload a file holding just the bytes `\r` and `\x01`; assess any applicant with it | The header was read with Python's `csv` module, which raises `csv.Error` on a bare `\r` inside a line; nothing caught it. Found by a Hypothesis property test feeding the reader random bytes | `cf3f1ac` | `tests/unit/test_statement.py::test_old_mac_line_endings_are_fine`<br>`tests/unit/test_statement.py::test_a_stray_carriage_return_is_refused_not_a_crash`<br>`tests/unit/test_properties.py::test_any_bytes_are_either_read_or_refused_with_reasons` |
| BUG-002 | 2026-10-19 | `/predict` too slow under load: p95 4.1 s at 10 users, against a 500 ms target | S3 Minor | Run `python perf/run_load_test.py` (10 users, each pausing 1 to 2 s, for 2 minutes) on the Windows laptop. Result: p50 3.0 s, p95 4.1 s, 2.7 requests a second, no failed requests | One app process handles every request, and one request costs about 230 ms of CPU (about 145 ms scoring, 85 ms web server and saving), so it finishes at most about 4 a second while the test asks for 5 to 6; the rest queue. Some work runs twice per request (the input check, and the model's preprocessing for scoring and again for the reasons) | Open | `tests/api/test_load_result.py::test_predict_met_the_local_speed_target` (strict expected failure) |
| BUG-003 | 2026-10-16 | On Windows the font was served with the wrong file type, so browsers refused it | S4 Cosmetic | On Windows, run `python -m pytest tests/unit/test_frontend.py`, or open any page with DevTools on the Network tab: the `.woff2` font doesn't arrive as `font/woff2`, and with `X-Content-Type-Options: nosniff` the browser drops it and falls back to a system font | Python's `mimetypes` reads file types from the Windows registry, where `.woff2` is often missing or wrong (and `.csv` is `application/vnd.ms-excel`). Fixed by setting every frontend file type in `main.py` (`STATIC_TYPES`) | `2c9a161` | `tests/unit/test_frontend.py::test_assets_are_served` |
| BUG-004 | 2026-10-19 | "Back to history" on a saved assessment lost the History page and filters | S3 Minor | Filter History (for example by reference), open an assessment, select "Back to history": History opens on page 1 with no filters | The link was built from the page the officer came from (`document.referrer`), which is always empty because the app sends `Referrer-Policy: no-referrer` (step 5.7). Fixed by History saving its own address in sessionStorage; only `/history.html` addresses are accepted | `28f4553` | `tests/e2e/test_journeys.py::test_the_back_link_returns_to_the_same_filtered_history_page`<br>`tests/unit/test_frontend.py::test_the_back_link_can_only_return_to_a_history_page` |
| BUG-005 | 2026-10-19 | No focus outline on the date boxes for keyboard users | S3 Minor | On New assessment, press Tab until the application date box: nothing shows where the focus is | Chrome moves focus into the day, month and year parts inside a date box, so the box itself never matches `:focus-visible`. Fixed with `:focus-within` | `54e4328` | `tests/unit/test_frontend.py::test_focus_is_always_visible` |
| BUG-006 | 2026-10-19 | A saved assessment scrolled sideways on a 320 px screen | S4 Cosmetic | Open a saved assessment 320 px wide (or a 1280 px screen at 400% zoom): the page is 1 px wider than the screen | The answers grid asked for columns of at least 18rem (288 px), more than a 320 px screen has left after padding. Every grid now asks for `min(18rem, 100%)` | `54e4328` | `tests/unit/test_frontend.py::test_grids_never_need_more_room_than_the_screen` |
| BUG-007 | 2026-10-19 | In Windows high-contrast mode the score marker's label couldn't be read | S3 Minor | Turn on a Windows contrast theme (or emulate forced colours in DevTools) and open a result: the marker's label is dark text on a black background | In forced-colours mode the browser replaces background colours, but the label kept its own text colour. Band tags, the scale and the marker now use system colours and keep visible edges | `54e4328` | `tests/unit/test_frontend.py::test_high_contrast_and_reduced_motion_are_handled` |
| BUG-008 | 2026-10-19 | The login page logged a console error for every signed-out visitor | S4 Cosmetic | Open `/` signed out with DevTools on the Console tab: a red 401 error for `/auth/me` (Lighthouse best practices 96) | The "already signed in?" check asked `/auth/me`, whose 401 is the right answer but shows as an error. Added `GET /auth/session`, which answers `{"user": null}` | `54e4328` | `tests/security/test_cookie_auth.py::test_session_check_never_401s`<br>`tests/unit/test_frontend.py::test_the_login_page_checks_the_session_without_a_401` |

## Summary

8 bugs: 7 fixed, 1 open (BUG-002, S3). None critical or major. By severity: 4 minor (S3) fixed
and 1 open, 3 cosmetic (S4) fixed.

| How it was found | Bugs |
|---|---|
| Hypothesis property test | BUG-001 |
| Load test (Locust) | BUG-002 |
| Unit test run on Windows | BUG-003 |
| Testing by hand in the browser | BUG-004 |
| Accessibility audit: keyboard tour, 320 px reflow, forced colours, Lighthouse | BUG-005 to BUG-008 |

## Notes on the open bug

**BUG-002.** Measured on the developer's Windows 11 laptop (8 CPUs, Python 3.12):

- one statement read in 8 ms;
- one scoring in about 145 ms;
- one `/predict` over HTTP, including saving, in about 230 ms.

Every request succeeded; they waited. Options weighed on 2026-11-03:

- remove the duplicated work (expected to cut scoring by about a third, with no score changing);
- run several app processes (the login rate limiter would first have to move into the
  database, and the free host has less than one CPU);
- model a realistic pace (10 officers pausing 10 to 30 s between assessments).

Recorded as a known defect for now. `test_load_result.py` keeps it visible and fails the day a
load test passes.
