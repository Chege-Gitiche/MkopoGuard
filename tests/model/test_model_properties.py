"""Step 7.4: properties of the shipped model, for ANY valid applicant (Hypothesis).

The golden examples and samples check fixed applicants. Here Hypothesis invents applicants
across every allowed answer, with and without a statement, and checks what must always hold:
a probability strictly between 0 and 1, the band and decision that follow from it, and exactly
3 raising and 2 lowering reasons, all different and in order of strength.
"""

from datetime import date

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mkopoguard import config
from mkopoguard.api import statement as reader
from mkopoguard.api.schemas import ApplicantProfile, band_for
from mkopoguard.api.service import score
from mkopoguard.bundle import load_bundle
from mkopoguard.explain.reasons import UNITS

pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / "manifest.json").exists(), reason="bundle not packaged yet"
)
APPLIED = date(2025, 6, 30)
MAYBE = st.sampled_from([True, False, None])
STATEMENT_CODES = {unit.code for unit in UNITS if unit.needs_statement}


@pytest.fixture(scope="module")
def champion():
    bundle = load_bundle()
    return bundle, bundle.reasoner()


@st.composite
def profiles(draw) -> ApplicantProfile:
    saved = draw(st.booleans())
    borrowed = draw(st.booleans())

    def follow_up(parent: bool):  # "Yes" only possible when the question before was "Yes"
        return draw(MAYBE if parent else st.sampled_from([False, None]))

    return ApplicantProfile.model_validate(
        {
            "applicant_ref": "PROP-" + draw(st.text("ABCDEFGH0123456789", min_size=1, max_size=8)),
            "application_date": APPLIED,
            "age": draw(st.integers(18, 100)),
            "income_quintile": draw(st.integers(1, 5)),
            "in_workforce": draw(st.booleans()),
            "education_level": draw(
                st.sampled_from(["primary_or_less", "secondary", "tertiary", None])
            ),
            "loan_amount_kes": draw(st.integers(1_000, 100_000)),
            "term_months": draw(st.sampled_from([1, 3, 6])),
            "saved_past_year": saved,
            "saved_in_savings_club": follow_up(saved),
            "borrowed_past_year": borrowed,
            "borrowed_from_fin_institution": follow_up(borrowed),
            "borrowed_from_family_friends": follow_up(borrowed),
            "borrowed_for_medical": follow_up(borrowed),
            "emergency_fund_source": draw(
                st.sampled_from(
                    [
                        "savings",
                        "work",
                        "family_friends",
                        "borrowing",
                        "selling_assets",
                        "other",
                        "could_not",
                        None,
                    ]
                )
            ),
            "worried_about_bills": draw(st.sampled_from(["very", "somewhat", "not_at_all", None])),
        }
    )


@st.composite
def statements(draw):
    """None (thin file) or a valid 6-month statement in the reader's format."""
    if draw(st.booleans()):
        return None
    n = draw(st.integers(reader.MIN_ROWS, 60))
    days = sorted(draw(st.lists(st.floats(1, 179), min_size=n - 1, max_size=n - 1)) + [170.0])
    balance, rows = draw(st.integers(0, 20_000)), []
    for day in reversed(days):
        kind = draw(st.sampled_from(sorted(reader.DIRECTION)))
        amount = draw(st.integers(1, 30_000))
        if reader.DIRECTION[kind] == "out" and amount > balance:
            kind = "income_in"
        balance += amount if reader.DIRECTION[kind] == "in" else -amount
        stamp = pd.Timestamp(APPLIED) - pd.Timedelta(seconds=round(day * 86_400))
        rows.append(f"{stamp:%Y-%m-%d %H:%M:%S},{kind},{reader.DIRECTION[kind]},{amount},{balance}")
    data = "\n".join([",".join(reader.COLUMNS), *rows]).encode("utf-8")
    return reader.parse_statement(data, APPLIED)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(profile=profiles(), statement=statements())
def test_every_valid_applicant_gets_a_complete_consistent_result(champion, profile, statement):
    bundle, reasoner = champion
    result = score(bundle, reasoner, profile, statement)
    p = result["probability_of_default"]
    assert 0 < p < 1
    assert result["band"] == band_for(p)
    assert result["decision"] == ("decline" if p >= result["threshold"] else "approve")
    raising, lowering = result["reasons"]["raising"], result["reasons"]["lowering"]
    assert len(raising) == 3 and len(lowering) == 2
    assert len({r["code"] for r in raising + lowering}) == 5
    for reason in raising + lowering:
        assert reason["direction"] == ("raises" if reason["impact_points"] > 0 else "lowers")
    # the strongest push up comes first, and every "raising" reason outranks every "lowering" one
    pushes = [r["impact_points"] for r in raising]
    assert pushes == sorted(pushes, reverse=True)
    assert min(pushes) >= max(r["impact_points"] for r in lowering) - 1e-9
    if statement is None:  # FR-13, for any thin-file applicant
        assert not {r["code"] for r in raising + lowering} & STATEMENT_CODES


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(profile=profiles(), statement=statements())
def test_the_score_is_repeatable_and_ignores_the_reference(champion, profile, statement):
    bundle, reasoner = champion
    first = score(bundle, reasoner, profile, statement)
    again = score(bundle, reasoner, profile, statement)
    renamed = profile.model_copy(update={"applicant_ref": "OTHER-REF"})
    other = score(bundle, reasoner, renamed, statement)
    assert first["probability_of_default"] == again["probability_of_default"]
    assert other["probability_of_default"] == first["probability_of_default"]
    assert other["reasons"] == first["reasons"]
