"""Step 1.6, part 1: give every applicant a loan (data card section 2.3).

Each simulation stage uses its own random stream, seeded from (SIMULATION_SEED, stage),
so changing one stage never shifts the random numbers of another.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

LOAN_STAGE = 1


def draw_amounts(income: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Loan = income x a log-normal ratio, rounded to KES 500 and clipped to the allowed range."""
    ratio = config.LOAN_RATIO_MEDIAN * np.exp(rng.normal(0, config.LOAN_RATIO_SIGMA, len(income)))
    ratio = np.clip(ratio, config.LOAN_RATIO_MIN, config.LOAN_RATIO_MAX)
    amount = np.round(income.to_numpy() * ratio / config.LOAN_ROUND_TO) * config.LOAN_ROUND_TO
    amount = np.clip(amount, config.LOAN_MIN, config.LOAN_MAX)
    return pd.Series(amount.astype(int), index=income.index)


def draw_terms(amount: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Pick 1, 3 or 6 months, with probabilities depending on the loan amount band."""
    u = rng.random(len(amount))  # one uniform number per applicant
    terms = np.zeros(len(amount), dtype=int)
    lower = 0
    for upper, probs in config.TERM_PROBS_BY_AMOUNT:
        in_band = (amount.to_numpy() > lower) & (amount.to_numpy() <= upper)
        cumulative = np.cumsum(probs)
        choice = np.searchsorted(cumulative, u[in_band], side="right")
        terms[in_band] = np.array(config.TERM_OPTIONS)[np.minimum(choice, len(probs) - 1)]
        lower = upper
    return pd.Series(terms, index=amount.index)


def add_loans(population: pd.DataFrame, seed: int = config.SIMULATION_SEED) -> pd.DataFrame:
    """Return a copy of the population with loan amount, term and true loan-to-income ratio."""
    rng = np.random.default_rng([seed, LOAN_STAGE])
    out = population.copy()
    income = out["hidden_monthly_income_kes"]

    out["loan_amount_kes"] = draw_amounts(income, rng)
    out["term_months"] = draw_terms(out["loan_amount_kes"], rng)
    # The ratio actually lent, after rounding and clipping. Hidden: it uses the true income.
    out["hidden_loan_to_income"] = out["loan_amount_kes"] / income
    return out
