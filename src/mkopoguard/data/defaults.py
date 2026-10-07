"""Step 1.6, part 3: decide who defaults (data card section 2.5).

risk = b0 + sum(weight x driver) + noise
p(default) = sigmoid(risk);  defaulted ~ Bernoulli(p)

The rule uses TRUE hidden values. The model will only ever see noisy traces of them in the
survey and the statement, so it can't predict perfectly. Gender and country are never used.
b0 is solved so the average default probability equals DEFAULT_TARGET_RATE.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

DEFAULT_STAGE = 4
NOISE_SD = 0.5

# Driver -> weight. Positive raises risk, negative lowers it. Copied into the data card.
WEIGHTS = {
    "log_loan_to_income": 0.6,  # borrowing more relative to income
    "pattern_seasonal": 0.3,  # compared with salaried income
    "pattern_irregular": 0.5,
    "stress": 0.5,
    "overdraft_propensity": 1.5,  # true overdraft dependence, 0-1
    "betting_share": 4.0,  # true share of income bet, 0-0.3
    "low_discipline": 0.8,  # = -discipline
    "saved_in_savings_club": -0.4,
    "saved_past_year": -0.3,
    "in_workforce": -0.3,
    "age_18_to_24": 0.2,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def risk_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """One column per driver in WEIGHTS. Missing survey answers count as 0."""
    pattern = df["hidden_income_pattern"].astype(str)
    yes = lambda col: (df[col] == 1).fillna(False).astype(float)  # noqa: E731
    return pd.DataFrame(
        {
            "log_loan_to_income": np.log(df["hidden_loan_to_income"]),
            "pattern_seasonal": (pattern == "seasonal").astype(float),
            "pattern_irregular": (pattern == "irregular").astype(float),
            "stress": df["hidden_stress"],
            "overdraft_propensity": df["hidden_overdraft_propensity"],
            "betting_share": df["hidden_betting_share"],
            "low_discipline": -df["hidden_discipline"],
            "saved_in_savings_club": yes("saved_in_savings_club"),
            "saved_past_year": yes("saved_past_year"),
            "in_workforce": yes("in_workforce"),
            "age_18_to_24": df["age"].between(18, 24).astype(float),
        },
        index=df.index,
    )


def solve_intercept(base: np.ndarray, target: float) -> float:
    """Find b0 so that mean(sigmoid(b0 + base)) == target, by bisection."""
    low, high = -20.0, 20.0
    for _ in range(100):
        mid = (low + high) / 2
        if _sigmoid(mid + base).mean() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def add_defaults(
    applicants: pd.DataFrame,
    seed: int = config.SIMULATION_SEED,
    target: float = config.DEFAULT_TARGET_RATE,
) -> pd.DataFrame:
    """Add the true default probability (hidden) and the defaulted label (0/1)."""
    rng = np.random.default_rng([seed, DEFAULT_STAGE])
    out = applicants.copy()

    drivers = risk_drivers(out)
    weighted = drivers.to_numpy() @ np.array([WEIGHTS[c] for c in drivers.columns])
    base = weighted + rng.normal(0, NOISE_SD, len(out))
    b0 = solve_intercept(base, target)

    probability = _sigmoid(b0 + base)
    out["hidden_default_probability"] = probability
    out["defaulted"] = (rng.random(len(out)) < probability).astype("int8")
    out.attrs["default_intercept"] = b0
    return out
