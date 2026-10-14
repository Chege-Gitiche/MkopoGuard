"""Step 5.4: the uploaded statement is checked thoroughly, and every problem is reported."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mkopoguard.api import statement as st
from mkopoguard.api.schemas import ApplicantProfile
from mkopoguard.features.transactions import FEATURES

APPLIED = date(2025, 6, 30)


def rows(days=175, start_balance=500):
    """A clean statement: monthly pay, daily airtime and shopping, balances that add up."""
    out, balance = [], start_balance
    first = datetime.combine(APPLIED, datetime.min.time()) - timedelta(days=days)
    for day in range(days):
        when = first + timedelta(days=day, hours=9)
        events = [("income_in", 15000)] if day % 30 == 0 else []
        events += [("airtime", 50), ("till", 120)]
        for minute, (kind, amount) in enumerate(events):
            balance += amount if st.DIRECTION[kind] == "in" else -amount
            stamp = (when + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")
            out.append([stamp, kind, st.DIRECTION[kind], str(amount), str(balance)])
    return out


def to_csv(table, header=st.COLUMNS) -> bytes:
    lines = [",".join(header)] + [",".join(r) for r in table]
    return ("\n".join(lines) + "\n").encode("utf-8")


def problems(data: bytes, applied=APPLIED) -> list[tuple[str, str]]:
    with pytest.raises(st.StatementError) as caught:
        st.parse_statement(data, applied)
    return caught.value.problems


def test_a_clean_statement_is_accepted():
    parsed = st.parse_statement(to_csv(rows()), APPLIED)
    assert len(parsed.transactions) == len(rows()) and parsed.warnings == []
    assert set(parsed.transactions["type"]) == {"income_in", "airtime", "till"}


def test_excels_utf8_with_bom_and_windows_line_endings_are_fine():
    data = b"\xef\xbb\xbf" + to_csv(rows()).replace(b"\n", b"\r\n")
    assert len(st.parse_statement(data, APPLIED).transactions) == len(rows())


def test_columns_in_another_order_are_fine():
    table = [[r[1], r[0], r[2], r[3], r[4]] for r in rows()]
    header = ("type", "timestamp", "direction", "amount_kes", "balance_after")
    assert len(st.parse_statement(to_csv(table, header), APPLIED).transactions) == len(table)


# --- the file itself -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b"x" * (st.MAX_BYTES + 1), "limit is 1 MB"),
        ("timestamp,type\n".encode("utf-16"), "UTF-8"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00", "UTF-8"),
    ],
    ids=["empty", "too big", "utf-16", "an image"],
)
def test_files_that_are_not_a_small_utf8_csv_are_refused(data, message):
    assert message in problems(data)[0][1]


def test_missing_extra_and_repeated_columns_are_all_named():
    header = ("timestamp", "type", "type", "amount", "balance_after", "note")
    found = " | ".join(what for _, what in problems(to_csv([], header)))
    assert "Missing column(s): direction, amount_kes" in found
    assert "Unexpected column(s): amount, note" in found
    assert "Repeated column(s): type" in found
    assert "timestamp,type,direction,amount_kes,balance_after" in found


def test_too_few_or_too_many_rows_are_refused():
    assert "at least 10" in problems(to_csv(rows()[:5]))[0][1]
    many = rows() * 30
    assert "limit is 10,000" in problems(to_csv(many))[0][1]


# --- every value ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column, value, message",
    [
        (0, "yesterday", "timestamp must be a date and time"),
        (1, "bribe", "type must be one of"),
        (2, "in", "direction doesn't match"),
        (3, "-50", "amount_kes must be a number above 0"),
        (3, "fifty", "amount_kes must be a number above 0"),
        (4, "-1", "balance_after must be a number, 0 or more"),
    ],
)
def test_a_bad_value_is_reported_with_its_row_number(column, value, message):
    table = rows()
    table[4][column] = value  # data row 5 = line 6 of the file
    found = problems(to_csv(table))
    assert any(where == "row 6" and message in what for where, what in found)


def test_every_bad_row_is_reported_and_the_list_is_capped():
    table = rows()
    for r in table[:30]:
        r[1] = "bribe"
    error = pytest.raises(st.StatementError, st.parse_statement, to_csv(table), APPLIED).value
    assert len(error.problems) == 30
    detail = error.detail()
    assert len(detail) == st.MAX_REPORTED + 1 and detail[-1]["msg"] == "and 10 more"
    assert detail[0]["loc"] == ["statement", "row 2"]


def test_an_edited_balance_is_caught():
    table = rows()
    table[20][3] = str(int(table[20][3]) * 10)  # someone made one payment look bigger
    found = problems(to_csv(table))
    assert found[0][0] == "row 22" and "doesn't add up" in found[0][1]


# --- the period it covers -----------------------------------------------------------------------


def test_transactions_on_or_after_the_application_date_are_refused():
    found = problems(to_csv(rows()), APPLIED - timedelta(5))
    assert "on or after the application date" in found[0][1]


def test_a_statement_that_is_too_short_is_refused():
    assert "covers only" in problems(to_csv(rows(days=100)))[0][1]


def test_an_old_statement_is_refused():
    assert "newest transaction is" in problems(to_csv(rows()), APPLIED + timedelta(90))[0][1]


def test_transactions_older_than_six_months_are_ignored_with_a_warning():
    parsed = st.parse_statement(to_csv(rows(days=200)), APPLIED)
    assert parsed.warnings and "older than 6 months were ignored" in parsed.warnings[0]


def test_times_with_a_timezone_are_converted_to_kenyan_time():
    table = rows()
    for r in table:
        r[0] = r[0].replace(" ", "T") + "+03:00"
    parsed = st.parse_statement(to_csv(table), APPLIED)
    assert parsed.transactions["timestamp"].iloc[0] == pd.Timestamp(rows()[0][0])


def test_garbage_never_crashes_the_parser():
    """Random bytes and random text: always a StatementError (a 422), never anything else."""
    rng = np.random.default_rng(0)
    clean = to_csv(rows())
    for i in range(200):
        if i % 2:
            data = rng.integers(0, 256, rng.integers(1, 3000)).astype(np.uint8).tobytes()
        else:  # a real statement with random characters overwritten
            buffer = bytearray(clean)
            for pos in rng.integers(0, len(buffer), rng.integers(1, 40)):
                buffer[pos] = int(rng.integers(32, 127))
            data = bytes(buffer)
        try:
            st.parse_statement(data, APPLIED)
        except st.StatementError:
            pass


# --- turning an application into model inputs ---------------------------------------------------


@pytest.fixture
def profile():
    return ApplicantProfile.model_validate(
        {
            "applicant_ref": "APP-1",
            "application_date": APPLIED,
            "age": 30,
            "income_quintile": 3,
            "in_workforce": True,
            "education_level": None,
            "loan_amount_kes": 15000,
            "term_months": 3,
            "saved_past_year": False,
            "saved_in_savings_club": None,
            "borrowed_past_year": False,
            "borrowed_from_fin_institution": None,
            "borrowed_from_family_friends": False,
            "borrowed_for_medical": False,
            "emergency_fund_source": "could_not",
            "worried_about_bills": "very",
        }
    )


def test_a_thin_file_application_has_no_statement_features(profile):
    inputs = st.build_inputs(profile, None)
    assert inputs.loc[0, "has_statement"] == 0
    assert inputs.loc[0, FEATURES].isna().all()


def test_a_statement_gives_every_statement_feature(profile):
    inputs = st.build_inputs(profile, st.parse_statement(to_csv(rows()), APPLIED))
    assert inputs.loc[0, "has_statement"] == 1
    assert inputs.loc[0, FEATURES].notna().all()
    assert inputs.loc[0, "stmt_avg_monthly_inflow"] == pytest.approx(15000, rel=0.01)
    assert inputs.loc[0, "stmt_loan_to_inflow"] == pytest.approx(1.0, rel=0.01)
