"""Survey statistics helpers.

Findex survey weights (survey_weight) make each country's sample represent that country's
adult population. They are only valid WITHIN one country, so every share here is computed
per country first. Pooled figures are the simple average of the 20 country shares.
"""

import pandas as pd


def weighted_share(
    df: pd.DataFrame,
    column: str,
    by: list[str],
    value=1,
    weight: str = "survey_weight",
) -> pd.Series:
    """Weighted share of respondents whose `column` equals `value`, for each group in `by`.

    Respondents with a missing answer are left out, like "don't know" in the survey.
    """
    d = df.loc[df[column].notna(), [*by, column, weight]]
    d = d.assign(_hits=(d[column] == value).astype(float) * d[weight])
    groups = d.groupby(by, observed=True)
    return groups["_hits"].sum() / groups[weight].sum()


def pool_average(by_country: pd.Series, country_level: str = "country_code") -> pd.Series:
    """Average the per-country shares, giving every country equal weight."""
    other_levels = [name for name in by_country.index.names if name != country_level]
    if not other_levels:
        return pd.Series({"pool": by_country.mean()})
    return by_country.groupby(level=other_levels, observed=True).mean()
