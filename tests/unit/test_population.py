import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.population import build_population, hidden_columns


def make_clean(n=1, **columns):
    """n cleaned respondents; pass a column=value or column=list to vary them."""
    data = {
        "country_code": "KEN",
        "respondent_id": list(range(1, n + 1)),
        "income_quintile": 3,
        "in_workforce": 1,
        "wage_payment_channel": "none",
        "agri_payment_channel": "none",
        "emergency_fund_source": "savings",
        "worried_about_bills": "not_at_all",
        "borrowed_for_medical": 0,
        "saved_past_year": 1,
        "has_mobile_money": 1,
    }
    data.update(columns)
    return pd.DataFrame(data, index=range(n))


def test_same_seed_gives_identical_output():
    clean = make_clean(200)
    pd.testing.assert_frame_equal(build_population(clean, seed=1), build_population(clean, seed=1))


def test_different_seed_gives_different_output():
    clean = make_clean(200)
    a = build_population(clean, seed=1)["hidden_monthly_income_kes"]
    b = build_population(clean, seed=2)["hidden_monthly_income_kes"]
    assert not a.equals(b)


def test_input_row_order_does_not_change_the_result():
    clean = make_clean(200)
    shuffled = clean.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(build_population(clean), build_population(shuffled))


def test_one_applicant_per_respondent_with_unique_ids():
    population = build_population(make_clean(50))
    assert len(population) == 50
    assert population["applicant_id"].is_unique
    assert population["applicant_id"].str.fullmatch(r"A-\d{5}").all()


@pytest.mark.parametrize(
    "wage, agri, expected",
    [
        ("account", "none", "salaried"),
        ("cash_only", "account", "salaried"),
        ("none", "cash_only", "seasonal"),
        ("other", "none", "irregular"),
        ("none", "none", "irregular"),
        (None, None, "irregular"),
    ],
)
def test_income_pattern_rules(wage, agri, expected):
    clean = make_clean(wage_payment_channel=wage, agri_payment_channel=agri)
    assert build_population(clean)["hidden_income_pattern"].iloc[0] == expected


def test_richer_quintiles_earn_more():
    clean = make_clean(5_000, income_quintile=[1, 2, 3, 4, 5] * 1_000)
    population = build_population(clean)
    medians = population.groupby("income_quintile")["hidden_monthly_income_kes"].median()
    assert medians.is_monotonic_increasing


def test_out_of_workforce_earns_less():
    clean = make_clean(2_000, in_workforce=[1, 0] * 1_000)
    population = build_population(clean)
    medians = population.groupby("in_workforce")["hidden_monthly_income_kes"].median()
    assert medians[0] < medians[1]


def test_fragile_applicants_have_higher_stress():
    fragile = {
        "emergency_fund_source": "could_not",
        "worried_about_bills": "very",
        "borrowed_for_medical": 1,
        "saved_past_year": 0,
    }
    clean = pd.concat([make_clean(500), make_clean(500, **fragile)], ignore_index=True)
    clean["respondent_id"] = range(1, 1_001)
    population = build_population(clean)
    is_fragile = population["emergency_fund_source"] == "could_not"
    stress = population["hidden_stress"]
    assert stress[is_fragile].mean() > stress[~is_fragile].mean()
    assert stress.mean() == pytest.approx(0, abs=1e-9)
    assert stress.std() == pytest.approx(1)


def test_application_dates_fall_within_the_configured_year():
    dates = build_population(make_clean(500))["application_date"]
    assert dates.min() >= pd.Timestamp(config.APPLICATION_START)
    assert dates.max() <= pd.Timestamp(config.APPLICATION_END)


def test_statement_window_is_180_days_before_application():
    population = build_population(make_clean(100))
    window = population["application_date"] - population["statement_start"]
    assert (window == pd.Timedelta(days=config.STATEMENT_DAYS)).all()


def test_has_statement_follows_mobile_money():
    population = build_population(make_clean(4, has_mobile_money=[1, 0, 1, 0]))
    assert population["has_statement"].tolist() == [1, 0, 1, 0]


def test_hidden_traits_are_marked_hidden():
    population = build_population(make_clean(10))
    assert hidden_columns(population) == [
        "hidden_monthly_income_kes",
        "hidden_income_pattern",
        "hidden_discipline",
        "hidden_stress",
    ]


def test_input_is_not_modified():
    clean = make_clean(20)
    before = clean.copy()
    build_population(clean)
    pd.testing.assert_frame_equal(clean, before)
