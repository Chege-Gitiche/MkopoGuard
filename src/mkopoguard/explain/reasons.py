"""Step 4.2: reason codes. Why did this applicant get this score, in plain English?

For one applicant: the 3 reasons pushing their risk up most and the 2 pushing it down most,
each with a stable code (for the API and history), a sentence a loan officer can read out,
and how many percentage points of risk it accounts for.

How the reasons are built from the exact SHAP contributions (step 4.1):
- Inputs that describe one thing are explained together ("units"): the three overdraft
  features are one reason, not three that may partly cancel each other out.
- THIN-FILE applicants: all statement features were filled in with averages, so the whole
  statement is ONE reason ("No M-Pesa statement provided"). A thin-file applicant is never
  told anything about betting, bills or overdraft: there is no statement to say it from.
- STATEMENT HOLDERS: each statement reason compares them with the typical statement holder
  (not with the average applicant, half of whom have no statement and only filled-in averages).
  What typical statement holders have in common goes into one reason, "Provided a 6-month
  M-Pesa statement". This only moves constants between reasons; the total is unchanged.
- Only the simulator's real drivers (explain/design.py) can be reasons. Non-drivers are added
  up as "other factors": their directions can be misleading once the drivers are in the model
  (e.g. more airtime spending LOWERS the score, though in the raw data it goes with MORE
  defaults), and a reason a loan officer can't make sense of is worse than none. The other
  factors' total is still reported, and flagged when it is as large as a stated reason.

Every unit's contribution plus "other factors" adds up exactly to the applicant's score
(log-odds) minus the base value, so nothing is hidden.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from mkopoguard.explain import design
from mkopoguard.explain.linear_shap import LinearShap, to_probability
from mkopoguard.features.groups import FEATURE_GROUPS

N_RAISING, N_LOWERING = 3, 2
MATERIAL_POINTS = 1.0  # other factors worth mentioning: at least 1 percentage point
STATEMENT_COLUMNS = FEATURE_GROUPS["statement"]
INCOME_GROUP = {1: "lowest", 2: "second-lowest", 3: "middle", 4: "second-highest", 5: "highest"}
EMERGENCY = {
    "could_not": "Could not raise emergency money",
    "savings": "Would use savings in an emergency",
    "work": "Would work extra to raise emergency money",
    "family_friends": "Would rely on family or friends in an emergency",
    "borrowing": "Would have to borrow in an emergency",
    "selling_assets": "Would have to sell assets in an emergency",
    "other": "Has another source of emergency money",
}
WORRY = {
    "very": "Very worried about paying monthly bills",
    "somewhat": "Somewhat worried about paying monthly bills",
    "not_at_all": "Not worried about paying monthly bills",
}


def _missing(value) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _yes_no(value, yes: str, no: str, unknown: str) -> str:
    if _missing(value):
        return unknown
    return yes if int(value) == 1 else no


def _ksh(value) -> str:
    return f"KSh {round(float(value), -2):,.0f}"


def _pct(value) -> str:
    return f"{float(value):.0%}"


# --- the reasons each unit can give ---------------------------------------------------------
# Each text function gets the applicant's raw values and whether the unit raises risk (up).


def _no_statement(v, up):
    return "No M-Pesa statement provided, so less is known about how money is managed"


def _statement(v, up):
    return "Provided a 6-month M-Pesa statement"


def _betting(v, up):
    share = v["stmt_betting_share"]
    if up:
        return f"Spends {_pct(share)} of outgoing money on betting"
    return "No spending on betting" if share < 0.005 else f"Low betting ({_pct(share)} of spending)"


def _overdraft(v, up):
    n, repaid = int(v["stmt_overdraft_borrow_count"]), v["stmt_overdraft_repay_ratio"]
    if n == 0:
        return "Did not use overdraft (Fuliza) in the last 6 months"
    if up:
        return f"Relies on overdraft (Fuliza): {n} times in 6 months, {_pct(repaid)} repaid"
    return f"Repays overdraft (Fuliza) reliably: {_pct(repaid)} of {n} borrowings repaid"


def _bills(v, up):
    months = int(v["stmt_paybill_months"])
    if months == 0:
        return "No bill payments (Paybill) on the statement"
    return f"Paid bills in {months} of the last 6 months"


def _income_level(v, up):
    amount = _ksh(v["stmt_avg_monthly_inflow"])
    if up:
        return f"Low money coming in: about {amount} a month"
    return f"Steady money coming in: about {amount} a month"


def _loan_to_income(v, up):
    ratio = float(v["stmt_loan_to_inflow"])
    if up:
        return f"Loan is {ratio:.1f} times monthly money coming in"
    return f"Loan is small next to income: {ratio:.1f} times monthly money coming in"


def _income_regularity(v, up):
    return "Income arrives irregularly" if up else "Income arrives regularly"


def _balance(v, up):
    low = v["stmt_low_balance_share"]
    if up:
        return f"Account is nearly empty {_pct(low)} of the time"
    return "Usually keeps money in the account"


def _loan_amount(v, up):
    amount = _ksh(v["loan_amount_kes"])
    return f"Large loan requested: {amount}" if up else f"Modest loan requested: {amount}"


def _income_quintile(v, up):
    q = v["income_quintile"]
    if _missing(q):
        return "Income group not known"
    return f"In the {INCOME_GROUP[int(q)]} income group in their country"


def _age(v, up):
    return f"Age {int(v['age'])}"


def _workforce(v, up):
    return _yes_no(
        v["in_workforce"], "In the workforce", "Not in the workforce", "Work status not known"
    )


def _saved(v, up):
    return _yes_no(
        v["saved_past_year"],
        "Saved money in the past year",
        "Did not save in the past year",
        "Did not say whether they saved",
    )


def _savings_group(v, up):
    return _yes_no(
        v["saved_in_savings_club"],
        "Saves with a savings group (chama)",
        "Does not save with a savings group (chama)",
        "Did not say whether they save with a savings group",
    )


def _medical(v, up):
    return _yes_no(
        v["borrowed_for_medical"],
        "Borrowed for medical costs in the past year",
        "Did not need to borrow for medical costs",
        "Did not say whether they borrowed for medical costs",
    )


def _emergency(v, up):
    value = v["emergency_fund_source"]
    if _missing(value) or str(value) not in EMERGENCY:
        return "Did not say where emergency money would come from"
    return EMERGENCY[str(value)]


def _worry(v, up):
    value = v["worried_about_bills"]
    if _missing(value) or str(value) not in WORRY:
        return "Did not say whether worried about paying bills"
    if str(value) == "somewhat" and not up:  # most applicants are "very" worried
        return "Only somewhat worried about paying monthly bills"
    return WORRY[str(value)]


@dataclass(frozen=True)
class Unit:
    code: str
    columns: tuple[str, ...]
    text: object  # function(values, up) -> str
    needs_statement: bool = False


UNITS = [
    Unit("betting", ("stmt_betting_share",), _betting, True),
    Unit(
        "overdraft",
        (
            "stmt_overdraft_borrow_count",
            "stmt_overdraft_repay_ratio",
            "stmt_overdraft_outstanding_months",
        ),
        _overdraft,
        True,
    ),
    Unit("bill_payments", ("stmt_paybill_months",), _bills, True),
    Unit(
        "income_level",
        ("stmt_avg_monthly_inflow", "stmt_median_monthly_inflow"),
        _income_level,
        True,
    ),
    Unit("loan_to_income", ("stmt_loan_to_inflow",), _loan_to_income, True),
    Unit(
        "income_regularity",
        ("stmt_inflow_volatility", "stmt_income_payments_per_month", "stmt_days_since_last_income"),
        _income_regularity,
        True,
    ),
    Unit("account_balance", ("stmt_end_balance_months", "stmt_low_balance_share"), _balance, True),
    Unit("statement_provided", (design.STATEMENT_FLAG,), _statement, True),
    Unit("loan_amount", ("loan_amount_kes",), _loan_amount),
    Unit("income_group", ("income_quintile",), _income_quintile),
    Unit("age", ("age",), _age),
    Unit("work_status", ("in_workforce",), _workforce),
    Unit("saved_past_year", ("saved_past_year",), _saved),
    Unit("savings_group", ("saved_in_savings_club",), _savings_group),
    Unit("medical_borrowing", ("borrowed_for_medical",), _medical),
    Unit("emergency_money", ("emergency_fund_source",), _emergency),
    Unit("bill_worry", ("worried_about_bills",), _worry),
]
NO_STATEMENT = Unit("no_statement", tuple(STATEMENT_COLUMNS), _no_statement)
BY_CODE = {u.code: u for u in [*UNITS, NO_STATEMENT]}
OTHER = "other_factors"


def statement_shift(explainer: LinearShap, background: pd.DataFrame) -> pd.Series:
    """For each statement column: how far the typical statement holder's contribution is from
    the average applicant's (log-odds). Used to compare holders with other holders."""
    holders = background[background[design.STATEMENT_FLAG] == 1]
    if holders.empty:
        raise ValueError("The background has no statement holders")
    typical = explainer.contributions(holders).mean()
    return typical[[c for c in STATEMENT_COLUMNS if c != design.STATEMENT_FLAG]]


def unit_contributions(
    contributions: pd.DataFrame, frame: pd.DataFrame, shift: pd.Series | None = None
) -> pd.DataFrame:
    """Contribution of every reason unit (log-odds); NaN where a unit doesn't apply.

    Columns: every unit code, "no_statement", and "other_factors" (all non-drivers).
    For every applicant, the non-missing values add up to the sum of all contributions.
    shift (from statement_shift) re-centres statement holders on the typical holder.
    """
    thin = (frame[design.STATEMENT_FLAG] == 0).to_numpy()
    contributions = contributions.copy()
    if shift is not None:
        holders = ~thin
        contributions.loc[holders, shift.index] -= shift.to_numpy()
        contributions.loc[holders, design.STATEMENT_FLAG] += float(shift.sum())
    out = pd.DataFrame(index=contributions.index)
    covered = set()
    for unit in UNITS:
        out[unit.code] = contributions[list(unit.columns)].sum(axis=1)
        covered |= set(unit.columns)
        if unit.needs_statement:
            out.loc[thin, unit.code] = np.nan
    out[NO_STATEMENT.code] = np.where(thin, contributions[STATEMENT_COLUMNS].sum(axis=1), np.nan)
    rest = [c for c in contributions if c not in covered]
    other = contributions[rest].sum(axis=1)
    statement_rest = [c for c in rest if c in STATEMENT_COLUMNS]
    other_thin = other - contributions[statement_rest].sum(axis=1)  # already in no_statement
    out[OTHER] = np.where(thin, other_thin, other)
    return out


@dataclass(frozen=True)
class Reason:
    code: str
    text: str
    direction: str  # "raises" or "lowers" the risk of default
    contribution: float  # log-odds
    impact_points: float  # percentage points of probability this reason accounts for
    strength: str  # "major" (5+ points), "moderate" (1-5) or "minor" (under 1)


def _impact(log_odds: float, contribution: float) -> float:
    """How much lower (or higher) the probability would be without this contribution."""
    return float(100 * (to_probability(log_odds) - to_probability(log_odds - contribution)))


def _strength(points: float) -> str:
    size = abs(points)
    return "major" if size >= 5 else "moderate" if size >= 1 else "minor"


def _reason(code: str, contribution: float, values: dict, log_odds: float) -> Reason:
    up = contribution > 0
    points = _impact(log_odds, contribution)
    text = BY_CODE[code].text(values, up)
    direction = "raises" if up else "lowers"
    return Reason(code, text, direction, float(contribution), points, _strength(points))


class Reasoner:
    """Builds reason codes for the champion. background = the development applicants."""

    def __init__(self, pipeline, background: pd.DataFrame):
        self.explainer = LinearShap(pipeline, background)
        self.shift = statement_shift(self.explainer, background)

    def units(self, frame: pd.DataFrame) -> pd.DataFrame:
        contribs = self.explainer.contributions(frame)
        return unit_contributions(contribs, frame, self.shift)

    def explain_rows(self, frame: pd.DataFrame) -> list[dict]:
        return explain_rows(self.explainer, frame, self.shift)

    def explain_applicant(self, applicant: pd.DataFrame) -> dict:
        return explain_applicant(self.explainer, applicant, self.shift)


def explain_rows(
    explainer: LinearShap, frame: pd.DataFrame, shift: pd.Series | None = None
) -> list[dict]:
    """Reason codes for every applicant in frame (one dict each)."""
    contribs = explainer.contributions(frame)
    units = unit_contributions(contribs, frame, shift)
    log_odds = explainer.log_odds(contribs)
    results = []
    for i, idx in enumerate(frame.index):
        row = units.loc[idx]
        values = frame.loc[idx].to_dict()
        candidates = row.drop(OTHER).dropna().sort_values(ascending=False)
        lo = float(log_odds.iloc[i])
        raising = [_reason(c, v, values, lo) for c, v in candidates.head(N_RAISING).items()]
        lowering_codes = candidates.iloc[::-1].drop(candidates.index[:N_RAISING], errors="ignore")
        lowering = [_reason(c, v, values, lo) for c, v in lowering_codes.head(N_LOWERING).items()]
        other = float(row[OTHER])
        other_impact = _impact(lo, other)
        third = raising[-1].contribution if raising else math.inf
        results.append(
            {
                "probability_of_default": float(to_probability(lo)),
                "base_probability": float(to_probability(explainer.base_value)),
                "raising": [asdict(r) for r in raising],
                "lowering": [asdict(r) for r in lowering],
                "other_factors": {
                    "contribution": other,
                    "impact_points": other_impact,
                    # as big as a stated reason (and at least 1 point): the UI mentions it
                    "material": bool(
                        abs(other) >= abs(third) and abs(other_impact) >= MATERIAL_POINTS
                    ),
                },
            }
        )
    return results


def explain_applicant(
    explainer: LinearShap, applicant: pd.DataFrame, shift: pd.Series | None = None
) -> dict:
    """Reason codes for exactly one applicant (a one-row table)."""
    if len(applicant) != 1:
        raise ValueError(f"explain_applicant takes one applicant, got {len(applicant)} rows")
    return explain_rows(explainer, applicant, shift)[0]
