"""Step 7.3: boundary values, tested on both sides of every limit.

Each limit gets the last value that must pass and the first that must fail (test cases
TC-17 to TC-22 and TC-26 to TC-29 in docs/test_plan.md). The other tests mostly check the
failing side; these make sure the limits themselves are where the requirements say, not one
step off.
"""

import json
from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from mkopoguard import config
from mkopoguard.api import statement as st
from mkopoguard.api.schemas import ApplicantProfile

SAMPLE = config.PROJECT_ROOT / "samples" / "profiles" / "sample_07_very_high.json"
APPLIED = date(2025, 6, 30)

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="samples not made yet (step 5.4)")


def profile(**changes) -> dict:
    values = json.loads(SAMPLE.read_text(encoding="utf-8"))
    values.update(changes)
    return values


def accepted(**changes) -> bool:
    try:
        ApplicantProfile.model_validate_json(json.dumps(profile(**changes)))
    except ValidationError:
        return False
    return True


# --- the applicant's answers --------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, last_ok, first_bad",
    [
        ("age", 18, 17),
        ("age", 100, 101),
        ("loan_amount_kes", 1_000, 999),
        ("loan_amount_kes", 100_000, 100_001),
        ("income_quintile", 1, 0),
        ("income_quintile", 5, 6),
    ],
)
def test_number_limits_are_inclusive(field, last_ok, first_bad):
    assert accepted(**{field: last_ok})
    assert not accepted(**{field: first_bad})


@pytest.mark.parametrize("months, ok", [(0, False), (1, True), (2, False), (3, True),
                                        (6, True), (12, False)])  # fmt: skip
def test_only_1_3_and_6_month_terms(months, ok):
    assert accepted(term_months=months) is ok


@pytest.mark.parametrize(
    "reference, ok",
    [("AB", False), ("ABC", True), ("A" * 64, True), ("A" * 65, False), ("123", False)],
)
def test_reference_length_and_needs_a_letter(reference, ok):
    assert accepted(applicant_ref=reference) is ok


def test_application_date_today_is_fine_tomorrow_is_not():
    today = date.today()
    assert accepted(application_date=today.isoformat())
    assert not accepted(application_date=(today + timedelta(days=1)).isoformat())


# --- the statement ----------------------------------------------------------------------------


def statement(days_before: list[float]) -> bytes:
    """One small airtime purchase at each of these many days before the application."""
    lines, balance = [",".join(st.COLUMNS)], 100_000
    applied = datetime.combine(APPLIED, datetime.min.time())
    for days in sorted(days_before, reverse=True):
        balance -= 10
        stamp = (applied - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{stamp},airtime,out,10,{balance}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def spread(n: int, oldest: float = 170, newest: float = 1) -> list[float]:
    step = (oldest - newest) / (n - 1)
    return [oldest - i * step for i in range(n)]


def problems(data: bytes) -> list[str]:
    try:
        st.parse_statement(data, APPLIED)
    except st.StatementError as error:
        return [what for _, what in error.problems]
    return []


def test_exactly_1_mb_is_accepted_one_byte_more_is_not():
    data = statement(spread(20))
    padded = data + b"\n" * (st.MAX_BYTES - len(data))  # blank lines are skipped
    assert len(padded) == st.MAX_BYTES and problems(padded) == []
    assert any("limit is 1 MB" in p for p in problems(padded + b"\n"))


@pytest.mark.parametrize("rows, ok", [(st.MIN_ROWS - 1, False), (st.MIN_ROWS, True),
                                      (st.MAX_ROWS, True), (st.MAX_ROWS + 1, False)])  # fmt: skip
def test_row_count_limits(rows, ok):
    assert (problems(statement(spread(rows))) == []) is ok


@pytest.mark.parametrize("oldest, ok", [(st.MIN_COVERAGE_DAYS, True),
                                        (st.MIN_COVERAGE_DAYS - 1, False)])  # fmt: skip
def test_the_statement_must_reach_back_150_days(oldest, ok):
    assert (problems(statement(spread(20, oldest=oldest))) == []) is ok


@pytest.mark.parametrize("newest, ok", [(st.MAX_DAYS_SINCE_LAST, True),
                                        (st.MAX_DAYS_SINCE_LAST + 1, False)])  # fmt: skip
def test_the_statement_must_run_to_within_60_days(newest, ok):
    assert (problems(statement(spread(20, newest=newest))) == []) is ok


def test_a_transaction_on_the_application_day_is_refused():
    assert any("on or after" in p for p in problems(statement(spread(20, newest=0))))
