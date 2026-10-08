"""Step 1.6, part 2: hidden spending behaviour and 6-month mobile money statements.

Data card sections 2.2 and 2.4. Two stages, each with its own random stream:
- add_behaviour():          hidden habits (overdrafts, betting, missed bills) for EVERY applicant
- generate_transactions():  a statement only for applicants with has_statement == 1

Every applicant's statement uses its own random stream, seeded from
(SIMULATION_SEED, TRANSACTION_STAGE, applicant number), so one applicant's statement
never depends on anyone else's.

Step 2.2 changes: a cash_out withdrawal after each inflow (most mobile money is withdrawn
as cash for daily spending), and an overdraft credit limit of one month's income.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

BEHAVIOUR_STAGE = 2
TRANSACTION_STAGE = 3

DAYS = config.STATEMENT_DAYS
MONTHS = DAYS // 30
SECONDS_PER_DAY = 86_400
LAST_SECOND = DAYS * SECONDS_PER_DAY - 10  # every transaction ends before the application date

OVERDRAFT_MAX_BORROW = 5_000  # KES per shortfall; larger ones can't be covered
OVERDRAFT_ROUND_TO = 100
OVERDRAFT_LIMIT_MONTHS = 1.0  # total overdraft owed can't exceed this many months of income

INFLOW_TYPES = {"income_in", "p2p_in", "overdraft_borrow"}
REPAY_TRIGGERS = {"income_in", "p2p_in"}  # money arriving gives a chance to repay overdrafts
REPAY_STRICTNESS = 4  # chance of repaying on each inflow = repay_share ** 4
CASH_TRIGGERS = {"income_in", "p2p_in"}  # money arriving is partly withdrawn as cash
CASH_SHARE_RANGE = (0.45, 0.75)  # share of each inflow withdrawn, drawn once per applicant

COLUMNS = ["applicant_id", "timestamp", "type", "direction", "amount_kes", "balance_after"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --- hidden behaviour ------------------------------------------------------


def add_behaviour(applicants: pd.DataFrame, seed: int = config.SIMULATION_SEED) -> pd.DataFrame:
    """Add hidden spending habits, driven by discipline and stress (data card 2.2 / 2.4)."""
    rng = np.random.default_rng([seed, BEHAVIOUR_STAGE])
    out = applicants.copy()
    n = len(out)
    discipline = out["hidden_discipline"].to_numpy()
    stress = out["hidden_stress"].to_numpy()

    # Overdraft habit: more likely with high stress and low discipline
    propensity = _sigmoid(-0.5 + 0.8 * stress - 0.6 * discipline)
    out["hidden_overdraft_propensity"] = propensity
    out["hidden_uses_overdraft"] = (rng.random(n) < propensity).astype("int8")
    out["hidden_overdraft_repay_share"] = _sigmoid(1.5 + 1.2 * discipline + rng.normal(0, 0.3, n))

    # Betting: low discipline makes someone more likely to bet, and to bet more
    is_bettor = rng.random(n) < _sigmoid(-0.9 - 0.9 * discipline)
    share = 0.02 + 0.28 * _sigmoid(-discipline + rng.normal(0, 0.5, n))
    out["hidden_betting_share"] = np.where(is_bettor, share, 0.0)

    # Chance of missing a monthly utility bill
    out["hidden_bill_miss_prob"] = _sigmoid(-2.0 - discipline)
    return out


# --- one applicant's statement ----------------------------------------------


def _times(rng, n, start_day=0.0, span_days=DAYS):
    """n random moments (in seconds since statement start) within a window of days."""
    seconds = (start_day + rng.uniform(0, span_days, n)) * SECONDS_PER_DAY
    return np.minimum(seconds.astype(np.int64), LAST_SECOND)


def _scheduled_events(a, rng):
    """All planned transactions for one applicant, before balances are applied."""
    income = a["hidden_monthly_income_kes"]
    times, kinds, amounts = [], [], []

    def add(t, kind, amt):
        times.append(np.atleast_1d(t))
        kinds.extend([kind] * len(np.atleast_1d(t)))
        amounts.append(np.atleast_1d(amt))

    # Income, shaped by the hidden income pattern
    pattern = a["hidden_income_pattern"]
    if pattern == "salaried":
        payday = rng.integers(0, 30)
        for m in range(MONTHS):
            add(
                _times(rng, 1, m * 30 + payday, 0.5),
                "income_in",
                income * (1 + rng.normal(0, 0.05)),
            )
    elif pattern == "seasonal":
        for q in range(MONTHS // 3):
            k = rng.integers(1, 3)
            total = 3 * income * rng.uniform(0.7, 1.3)
            add(_times(rng, k, q * 90, 90), "income_in", total * rng.dirichlet(np.ones(k)))
    else:  # irregular
        for m in range(MONTHS):
            k = rng.integers(8, 21)
            total = income * max(0.2, 1 + rng.normal(0, 0.4))
            add(_times(rng, k, m * 30, 30), "income_in", total * rng.dirichlet(np.full(k, 2.0)))

    # Money from and to family and friends
    if a["received_domestic_remittance"] == 1:
        k = rng.poisson(MONTHS)
        add(_times(rng, k), "p2p_in", income * rng.uniform(0.05, 0.20, k))
    if a["sent_domestic_remittance"] == 1:
        k = rng.integers(MONTHS, 3 * MONTHS + 1)
        add(_times(rng, k), "p2p_out", income * rng.uniform(0.05, 0.15, k))

    # Utility bill via Paybill, sometimes missed
    if a["utility_payment_channel"] == "account":
        bill_day = rng.integers(0, 30)
        bill = income * rng.uniform(0.03, 0.10)
        for m in range(MONTHS):
            if rng.random() >= a["hidden_bill_miss_prob"]:
                add(
                    _times(rng, 1, m * 30 + bill_day, 1),
                    "paybill",
                    bill * (1 + rng.normal(0, 0.05)),
                )

    # Shop payments via Till
    if a["made_digital_merchant_payment"] == 1:
        k = rng.integers(5 * MONTHS, 30 * MONTHS + 1)
        add(_times(rng, k), "till", income * rng.uniform(0.005, 0.03, k))

    # Airtime: everyone with a statement, KES 20-200
    k = rng.integers(4 * MONTHS, 15 * MONTHS + 1)
    add(_times(rng, k), "airtime", rng.integers(2, 21, k) * 10)

    # Betting
    if a["hidden_betting_share"] > 0:
        k = rng.integers(4 * MONTHS, 20 * MONTHS + 1)
        total = a["hidden_betting_share"] * income * MONTHS
        add(_times(rng, k), "betting", total * rng.dirichlet(np.ones(k)))

    t = np.concatenate(times)
    amt = np.maximum(np.round(np.concatenate(amounts)), 1).astype(np.int64)
    return t, np.array(kinds), amt


def statement_for(a, seed: int = config.SIMULATION_SEED) -> pd.DataFrame:
    """Generate one applicant's statement, applying balances and overdraft rules in time order."""
    number = int(a["applicant_id"].split("-")[1])
    rng = np.random.default_rng([seed, TRANSACTION_STAGE, number])

    balance = float(round(a["hidden_monthly_income_kes"] * rng.uniform(0, 0.5)))  # opening balance
    cash_share = rng.uniform(*CASH_SHARE_RANGE)
    t, kinds, amounts = _scheduled_events(a, rng)
    order = np.argsort(t, kind="stable")

    uses_overdraft = a["hidden_uses_overdraft"] == 1
    repay_share = a["hidden_overdraft_repay_share"]
    limit = OVERDRAFT_LIMIT_MONTHS * a["hidden_monthly_income_kes"]
    owed = 0
    rows = []  # (seconds, type, direction, amount, balance_after)

    for i in order:
        sec, kind, amt = int(t[i]), kinds[i], int(amounts[i])
        if kind in INFLOW_TYPES:
            balance += amt
            rows.append((sec, kind, "in", amt, balance))
            # Low-discipline applicants repay less often, and never more than half their balance
            if owed > 0 and kind in REPAY_TRIGGERS and rng.random() < repay_share**REPAY_STRICTNESS:
                repay = int(min(owed, balance // 2))
                if repay > 0:
                    balance -= repay
                    owed -= repay
                    rows.append(
                        (min(sec + 1, LAST_SECOND), "overdraft_repay", "out", repay, balance)
                    )
            # Part of each inflow is withdrawn as cash for daily spending
            if kind in CASH_TRIGGERS:
                cash = int(min(round(amt * cash_share, -1), balance))
                if cash > 0:
                    balance -= cash
                    rows.append((min(sec + 2, LAST_SECOND), "cash_out", "out", cash, balance))
            continue

        shortfall = amt - balance
        borrow = int(np.ceil(shortfall / OVERDRAFT_ROUND_TO) * OVERDRAFT_ROUND_TO)
        if amt <= balance:
            balance -= amt
            rows.append((sec, kind, "out", amt, balance))
        elif uses_overdraft and shortfall <= OVERDRAFT_MAX_BORROW and owed + borrow <= limit:
            balance += borrow
            owed += borrow
            rows.append((sec, "overdraft_borrow", "in", borrow, balance))
            balance -= amt
            rows.append((min(sec + 1, LAST_SECOND), kind, "out", amt, balance))
        # otherwise: not enough money and no overdraft room, so the payment doesn't happen

    stmt = pd.DataFrame(
        rows, columns=["seconds", "type", "direction", "amount_kes", "balance_after"]
    )
    stmt.insert(0, "applicant_id", a["applicant_id"])
    stmt.insert(
        1, "timestamp", a["statement_start"] + pd.to_timedelta(stmt.pop("seconds"), unit="s")
    )
    stmt["balance_after"] = stmt["balance_after"].round().astype(np.int64)
    return stmt[COLUMNS]


def generate_transactions(
    applicants: pd.DataFrame, seed: int = config.SIMULATION_SEED
) -> pd.DataFrame:
    """Statements for every applicant with has_statement == 1, as one long table."""
    with_statement = applicants[applicants["has_statement"] == 1]
    statements = [statement_for(a, seed) for a in with_statement.to_dict("records")]
    out = pd.concat(statements, ignore_index=True)
    out["type"] = out["type"].astype("category")
    out["direction"] = out["direction"].astype("category")
    return out
