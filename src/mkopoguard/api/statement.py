"""Step 5.4: reading an uploaded M-Pesa statement, and turning an application into model inputs.

The statement is a CSV in one fixed format, oldest transaction first:

    timestamp,type,direction,amount_kes,balance_after
    2025-03-01 08:16:35,income_in,in,12000,12950
    2025-03-01 09:02:10,airtime,out,50,12900

It is checked before anything is computed from it, and every problem is reported with its row
number, so the officer can fix the file:
- a CSV file, UTF-8 text, at most 1 MB and 10,000 rows, with exactly these five columns;
- every timestamp a date and time, every type one of the 10 known types with its direction,
  amounts positive, balances not negative;
- the balance adds up from row to row (a statement that doesn't add up has been edited);
- it covers the 6 months before the application date: nothing on or after that date, the
  oldest transaction at least 150 days before it, the newest within 60 days of it.
  (Every simulated statement the model was trained on meets these rules.)

The features are then built by the SAME Phase 2 code that built the training data
(features/transactions.py), so the model sees uploaded statements exactly as it saw training.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.api.schemas import ApplicantProfile
from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES
from mkopoguard.features.transactions import statement_features

COLUMNS = ("timestamp", "type", "direction", "amount_kes", "balance_after")
DIRECTION = {
    "income_in": "in",
    "p2p_in": "in",
    "overdraft_borrow": "in",
    "p2p_out": "out",
    "cash_out": "out",
    "till": "out",
    "paybill": "out",
    "airtime": "out",
    "betting": "out",
    "overdraft_repay": "out",
}
MAX_BYTES = 1_048_576  # 1 MB
MAX_ROWS = 10_000
MIN_ROWS = 10
MIN_COVERAGE_DAYS = 150
MAX_DAYS_SINCE_LAST = 60
MAX_REPORTED = 20  # problems listed before "and N more"
APPLICANT = "applicant"  # the one applicant id used inside the feature code


class StatementError(ValueError):
    """The statement can't be used. problems: (where, what) pairs, e.g. ("row 12", "...")."""

    def __init__(self, problems: list[tuple[str, str]]):
        self.problems = problems
        super().__init__("; ".join(f"{where}: {what}" for where, what in problems))

    def detail(self) -> list[dict]:
        """In the same shape as FastAPI's own 422 errors."""
        shown = [
            {"loc": ["statement", where], "msg": what, "type": "statement_invalid"}
            for where, what in self.problems[:MAX_REPORTED]
        ]
        if len(self.problems) > MAX_REPORTED:
            more = len(self.problems) - MAX_REPORTED
            shown.append(
                {"loc": ["statement"], "msg": f"and {more} more", "type": "statement_invalid"}
            )
        return shown


@dataclass
class Statement:
    transactions: pd.DataFrame  # the five columns, parsed, plus applicant_id
    warnings: list[str] = field(default_factory=list)


def _fail(where: str, what: str):
    raise StatementError([(where, what)])


def _read_table(data: bytes) -> pd.DataFrame:
    if not data:
        _fail("file", "The file is empty")
    if len(data) > MAX_BYTES:
        _fail("file", f"The file is {len(data) / 1_048_576:.1f} MB; the limit is 1 MB")
    try:
        text = data.decode("utf-8-sig")  # -sig: also accepts Excel's "UTF-8 with BOM"
    except UnicodeDecodeError:
        _fail("file", "The file isn't UTF-8 text; save it as CSV (UTF-8)")
    if "\x00" in text:
        _fail("file", "The file isn't a text CSV")
    header = next(csv.reader(io.StringIO(text)), [])
    names = [h.strip() for h in header]
    missing = [c for c in COLUMNS if c not in names]
    extra = [c for c in names if c not in COLUMNS]
    repeated = sorted({c for c in names if names.count(c) > 1})
    problems = []
    if missing:
        problems.append(("columns", f"Missing column(s): {', '.join(missing)}"))
    if extra:
        problems.append(("columns", f"Unexpected column(s): {', '.join(extra)}"))
    if repeated:
        problems.append(("columns", f"Repeated column(s): {', '.join(repeated)}"))
    if problems:
        problems.append(("columns", f"The header must be exactly: {','.join(COLUMNS)}"))
        raise StatementError(problems)
    try:
        table = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        _fail("file", f"Not a valid CSV: {str(error).splitlines()[0]}")
    table.columns = [c.strip() for c in table.columns]
    if len(table) < MIN_ROWS:
        _fail(
            "file", f"Only {len(table)} transactions; a 6-month statement has at least {MIN_ROWS}"
        )
    if len(table) > MAX_ROWS:
        _fail("file", f"{len(table):,} transactions; the limit is {MAX_ROWS:,}")
    return table.apply(lambda col: col.str.strip())


def _parse_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Parse every value; report each bad one with the row number a spreadsheet would show."""
    row = pd.Series([f"row {i + 2}" for i in range(len(table))], index=table.index)  # header = 1
    problems: list[tuple[str, str]] = []

    def bad(mask: pd.Series, message: str) -> None:
        problems.extend((where, message) for where in row[mask])

    try:
        timestamp = pd.to_datetime(table["timestamp"], format="ISO8601", errors="coerce")
    except (ValueError, TypeError):  # e.g. some times with a timezone and some without
        _fail("timestamp", "All timestamps must be in the same format, e.g. 2025-03-01 08:16:35")
    if timestamp.dt.tz is not None:  # an explicit offset: convert to Kenyan time
        timestamp = timestamp.dt.tz_convert("Africa/Nairobi").dt.tz_localize(None)
    bad(timestamp.isna(), "timestamp must be a date and time, e.g. 2025-03-01 08:16:35")

    kind = table["type"].str.lower()
    bad(~kind.isin(DIRECTION), f"type must be one of: {', '.join(DIRECTION)}")
    direction = table["direction"].str.lower()
    expected = kind.map(DIRECTION)
    bad(kind.isin(DIRECTION) & (direction != expected), "direction doesn't match the type")

    amount = pd.to_numeric(table["amount_kes"], errors="coerce")
    bad(~(amount > 0), "amount_kes must be a number above 0")
    balance = pd.to_numeric(table["balance_after"], errors="coerce")
    bad(~(balance >= 0), "balance_after must be a number, 0 or more")
    if problems:
        raise StatementError(problems)

    # The balance must add up: each balance = the one before +/- this amount (file order)
    signed = np.where(direction == "in", amount, -amount)
    off = (balance.diff() - signed).abs() > 0.005
    off.iloc[0] = False
    if off.any():
        rows = list(row[off][:3])
        raise StatementError(
            [(r, "balance_after doesn't add up with the row before it") for r in rows]
            + [("balance", "The balances don't add up: the statement may have been edited")]
        )
    return pd.DataFrame(
        {
            "applicant_id": APPLICANT,
            "timestamp": timestamp,
            "type": kind,
            "direction": direction,
            "amount_kes": amount,
            "balance_after": balance,
        }
    )


def parse_statement(data: bytes, application_date: date) -> Statement:
    """Check an uploaded statement and parse it. Raises StatementError listing every problem."""
    transactions = _parse_rows(_read_table(data))
    applied = pd.Timestamp(application_date)
    days_before = (applied - transactions["timestamp"]).dt.total_seconds() / 86_400
    problems = []
    future = int((days_before <= 0).sum())
    if future:
        problems.append(("period", f"{future} transaction(s) on or after the application date"))
    oldest, newest = days_before.max(), days_before.min()
    if oldest < MIN_COVERAGE_DAYS:
        problems.append(
            (
                "period",
                f"The statement covers only {oldest:.0f} days before the application; "
                f"it needs at least {MIN_COVERAGE_DAYS} of the 6 months",
            )
        )
    if newest > MAX_DAYS_SINCE_LAST and not future:
        problems.append(
            (
                "period",
                f"The newest transaction is {newest:.0f} days before the application; "
                f"the statement must run to within {MAX_DAYS_SINCE_LAST} days of it",
            )
        )
    if problems:
        raise StatementError(problems)
    warnings = []
    old = int((days_before > config.STATEMENT_DAYS).sum())
    if old:
        warnings.append(f"{old} transaction(s) older than 6 months were ignored")
    return Statement(transactions, warnings)


def build_inputs(profile: ApplicantProfile, statement: Statement | None) -> pd.DataFrame:
    """One row of model inputs: the profile's answers plus the statement's features."""
    row = profile.model_inputs()
    row["has_statement"] = int(statement is not None)
    if statement is None:
        row |= dict.fromkeys(STATEMENT_FEATURES, np.nan)
    else:
        applicant = pd.DataFrame(
            {
                "applicant_id": [APPLICANT],
                "application_date": [pd.Timestamp(profile.application_date)],
                "loan_amount_kes": [profile.loan_amount_kes],
            }
        )
        features = statement_features(statement.transactions, applicant)
        row |= features.loc[APPLICANT].to_dict()
    return pd.DataFrame([row])
