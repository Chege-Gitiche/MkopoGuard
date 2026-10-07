"""Step 1.5: turn cleaned survey respondents into simulated loan applicants.

Each adult respondent becomes exactly one applicant. On top of their real survey
answers, every applicant gets hidden traits (see docs/data_card.md, section 2.2).
Hidden columns start with "hidden_" and must NEVER be used as model features.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

HIDDEN_PREFIX = "hidden_"

# Weights for the stress score (financial fragility); missing answers count as 0
STRESS_WEIGHTS = {
    "could_not_find_emergency_funds": 1.0,
    "very_worried_about_bills": 0.7,
    "borrowed_for_medical": 0.5,
    "did_not_save": 0.5,
}
STRESS_NOISE_SD = 0.5


def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into a fixed order and give each applicant an ID like A-00001."""
    out = df.sort_values(["country_code", "respondent_id"]).reset_index(drop=True)
    out.insert(0, "applicant_id", [f"A-{i:05d}" for i in range(1, len(out) + 1)])
    return out


def draw_income(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Log-normal monthly income around the quintile median, lower if out of workforce."""
    median = df["income_quintile"].astype(int).map(config.INCOME_MEDIAN_BY_QUINTILE)
    factor = np.where(df["in_workforce"] == 1, 1.0, config.OUT_OF_WORKFORCE_INCOME_FACTOR)
    noise = rng.normal(0, config.INCOME_SIGMA, size=len(df))
    income = median.to_numpy() * factor * np.exp(noise)
    return pd.Series(np.round(income, -1), index=df.index)  # nearest KES 10


def income_pattern(df: pd.DataFrame) -> pd.Series:
    """salaried if paid wages; else seasonal if paid for farm goods; else irregular."""
    wages = df["wage_payment_channel"].isin(["account", "cash_only"])
    farm = df["agri_payment_channel"].isin(["account", "cash_only", "other"])
    pattern = np.select([wages, farm], ["salaried", "seasonal"], default="irregular")
    return pd.Series(pattern, index=df.index, dtype="category")


def stress_score(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Weighted fragility signals plus noise, scaled to mean 0 and standard deviation 1."""
    signals = {
        "could_not_find_emergency_funds": df["emergency_fund_source"] == "could_not",
        "very_worried_about_bills": df["worried_about_bills"] == "very",
        "borrowed_for_medical": df["borrowed_for_medical"] == 1,
        "did_not_save": df["saved_past_year"] == 0,
    }
    raw = sum(STRESS_WEIGHTS[name] * s.fillna(False).astype(float) for name, s in signals.items())
    raw = raw + rng.normal(0, STRESS_NOISE_SD, size=len(df))
    return (raw - raw.mean()) / raw.std()


def application_dates(n: int, rng: np.random.Generator) -> pd.Series:
    """A random application date within the configured year."""
    start = pd.Timestamp(config.APPLICATION_START)
    days = (pd.Timestamp(config.APPLICATION_END) - start).days + 1
    return start + pd.to_timedelta(rng.integers(0, days, size=n), unit="D")


def build_population(clean: pd.DataFrame, seed: int = config.SIMULATION_SEED) -> pd.DataFrame:
    """One applicant per cleaned respondent, with hidden traits and dates.

    The same input and seed always give an identical result.
    """
    rng = np.random.default_rng(seed)
    out = assign_ids(clean)

    # Random draws always happen in this fixed order, so results are reproducible
    out["hidden_monthly_income_kes"] = draw_income(out, rng)
    out["hidden_income_pattern"] = income_pattern(out)
    out["hidden_discipline"] = rng.normal(0, 1, size=len(out))
    out["hidden_stress"] = stress_score(out, rng)
    out["application_date"] = application_dates(len(out), rng)

    out["statement_start"] = out["application_date"] - pd.Timedelta(days=config.STATEMENT_DAYS)
    out["has_statement"] = (out["has_mobile_money"] == 1).astype("int8")
    return out


def hidden_columns(df: pd.DataFrame) -> list[str]:
    """Names of the columns the model must never see."""
    return [c for c in df.columns if c.startswith(HIDDEN_PREFIX)]
