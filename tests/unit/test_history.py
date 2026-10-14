"""Step 5.6: each officer sees only their own history, newest first, paginated and filtered."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api import db, security
from mkopoguard.api.history import MAX_PAGE_SIZE, NOT_FOUND
from mkopoguard.api.main import create_app
from mkopoguard.api.models import Prediction
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import create_user, find_user

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
SAMPLES = config.PROJECT_ROOT / "samples"
START = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
REASONS = {
    "raising": [
        {"code": c, "text": c, "direction": "raises", "impact_points": 3.0, "strength": "moderate"}
        for c in ("betting", "bill_worry", "loan_amount")
    ],
    "lowering": [
        {"code": c, "text": c, "direction": "lowers", "impact_points": -2.0, "strength": "moderate"}
        for c in ("saved_past_year", "work_status")
    ],
    "other_factors": {"impact_points": 0.1, "material": False},
}
BANDS = ("low", "medium", "high", "very_high")


def stored(officer_id: int, i: int, **overrides) -> Prediction:
    values = {
        "officer_id": officer_id,
        "applicant_ref": f"APP-{i:03d}",
        "inputs": {"profile": {"term_months": 3}, "statement_features": None},
        "probability": 0.05 + 0.01 * (i % 25),
        "band": BANDS[i % 4],
        "decision": "approve" if i % 2 else "decline",
        "threshold": 0.196,
        "reasons": REASONS,
        "model_version": config.MODEL_VERSION,
        "created_at": START + timedelta(hours=i),
    }
    return Prediction(**values | overrides)


@pytest.fixture(scope="module")
def url(tmp_path_factory):
    """Officer A has 25 predictions, officer B has 3, the demo account none."""
    url = f"sqlite:///{(tmp_path_factory.mktemp('history') / 'h.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    security.BCRYPT_ROUNDS, rounds = 4, security.BCRYPT_ROUNDS
    with db.make_sessions(engine)() as session:
        for email, role in (("a@bank.example", "officer"), ("b@bank.example", "officer"),
                            ("demo@bank.example", "demo")):  # fmt: skip
            create_user(session, email, email[0].upper(), PASSWORD, role)
        a, b = find_user(session, "a@bank.example"), find_user(session, "b@bank.example")
        session.add_all([stored(a.id, i) for i in range(25)])
        session.add_all([stored(b.id, 100 + i, applicant_ref=f"B-{i}") for i in range(3)])
        session.commit()
    security.BCRYPT_ROUNDS = rounds
    engine.dispose()
    return url


@pytest.fixture(scope="module")
def client(url):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


def login(client, email):
    token = client.post("/auth/login", data={"username": email, "password": PASSWORD})
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


@pytest.fixture(scope="module")
def a(client):
    return login(client, "a@bank.example")


@pytest.fixture(scope="module")
def b(client):
    return login(client, "b@bank.example")


def history(client, headers, **params):
    response = client.get("/predictions", headers=headers, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def ids_of(client, headers, **params) -> list[int]:
    return [item["id"] for item in history(client, headers, **params)["items"]]


def b_prediction_id(client, b) -> int:
    return history(client, b)["items"][0]["id"]


# --- only your own ------------------------------------------------------------------------------


def test_officer_a_gets_404_on_officer_bs_prediction(client, a, b):
    """The guide's done-when test."""
    theirs = b_prediction_id(client, b)
    assert client.get(f"/predictions/{theirs}", headers=b).status_code == 200  # it exists
    response = client.get(f"/predictions/{theirs}", headers=a)
    assert response.status_code == 404
    assert response.json() == {"detail": NOT_FOUND}


def test_someone_elses_prediction_looks_exactly_like_a_missing_one(client, a, b):
    theirs = client.get(f"/predictions/{b_prediction_id(client, b)}", headers=a)
    missing = client.get("/predictions/999999", headers=a)
    assert (theirs.status_code, theirs.json()) == (missing.status_code, missing.json())


def test_the_list_never_includes_anyone_elses_predictions(client, a, b):
    every_a = history(client, a, size=MAX_PAGE_SIZE)
    assert every_a["total"] == 25
    assert not any(item["applicant_ref"].startswith("B-") for item in every_a["items"])
    assert history(client, b)["total"] == 3


def test_the_demo_account_has_an_empty_history(client):
    demo = login(client, "demo@bank.example")
    assert history(client, demo) == {"items": [], "page": 1, "size": 20, "total": 0, "pages": 0}


def test_history_needs_a_login(client):
    assert client.get("/predictions").status_code == 401
    assert client.get("/predictions/1").status_code == 401


# --- order and pages ----------------------------------------------------------------------------


def test_newest_first(client, a):
    items = history(client, a, size=MAX_PAGE_SIZE)["items"]
    times = [item["created_at"] for item in items]
    assert times == sorted(times, reverse=True)
    assert items[0]["applicant_ref"] == "APP-024"


def test_pages_cover_everything_once(client, a):
    first = history(client, a, page=1, size=10)
    assert (first["total"], first["pages"], len(first["items"])) == (25, 3, 10)
    seen = [i for page in (1, 2, 3) for i in ids_of(client, a, page=page, size=10)]
    assert len(seen) == 25 and len(set(seen)) == 25
    assert ids_of(client, a, page=3, size=10) and len(ids_of(client, a, page=3, size=10)) == 5


def test_a_page_past_the_end_is_empty_not_an_error(client, a):
    beyond = history(client, a, page=99, size=10)
    assert beyond["items"] == [] and beyond["total"] == 25


@pytest.mark.parametrize(
    "params",
    [{"page": 0}, {"size": 0}, {"size": MAX_PAGE_SIZE + 1}, {"page": "two"}, {"band": "huge"},
     {"decision": "maybe"}, {"date_from": "14/10/2026"}],
)  # fmt: skip
def test_bad_query_parameters_are_a_422(client, a, params):
    assert client.get("/predictions", headers=a, params=params).status_code == 422


def test_a_non_numeric_id_is_a_422(client, a):
    assert client.get("/predictions/abc", headers=a).status_code == 422


# --- filters ------------------------------------------------------------------------------------


def test_filter_by_band_and_decision(client, a):
    high = history(client, a, band="high", size=MAX_PAGE_SIZE)
    assert high["total"] == 6 and {i["band"] for i in high["items"]} == {"high"}
    both = history(client, a, band="high", decision="decline", size=MAX_PAGE_SIZE)
    assert {(i["band"], i["decision"]) for i in both["items"]} == {("high", "decline")}


def test_filter_by_reference_prefix_ignores_case_and_wildcards(client, a):
    assert history(client, a, ref="app-02")["total"] == 5  # APP-020 .. APP-024
    assert history(client, a, ref="APP-%")["total"] == 0  # % is a character, not a wildcard


def test_filter_by_kenyan_calendar_days(client, a):
    # APP-000 is at 09:00 UTC on 1 Oct = 12:00 in Nairobi; APP-015 is 1 Oct 24:00 UTC
    # = 03:00 on 2 Oct in Nairobi. So 1 Oct (Kenyan day) holds APP-000 .. APP-011.
    first_day = history(client, a, date_from="2026-10-01", date_to="2026-10-01", size=100)
    refs = sorted(i["applicant_ref"] for i in first_day["items"])
    assert refs == [f"APP-{i:03d}" for i in range(12)]
    assert history(client, a, date_from="2026-10-02", size=100)["total"] == 13


def test_date_from_after_date_to_is_refused(client, a):
    response = client.get(
        "/predictions", headers=a, params={"date_from": "2026-10-05", "date_to": "2026-10-01"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "date_from"]


# --- one prediction in full -----------------------------------------------------------------------


def test_a_prediction_made_through_the_api_comes_back_in_full(client, a):
    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    s = next(x for x in manifest if x["statement"])
    files = {"statement": ("s.csv", (SAMPLES / s["statement"]).read_bytes(), "text/csv")}
    made = client.post(
        "/predict", headers=a, data={"profile": (SAMPLES / s["profile"]).read_text()}, files=files
    ).json()
    detail = client.get(f"/predictions/{made['id']}", headers=a).json()
    assert detail["probability_of_default"] == pytest.approx(made["probability_of_default"])
    assert detail["threshold"] == pytest.approx(made["threshold"])
    assert detail["reasons"] == made["reasons"]
    assert detail["profile"]["applicant_ref"] == made["applicant_ref"]
    assert detail["has_statement"] is True and detail["statement_transactions"] > 0
    assert detail["term_months"] == made["term_months"]
    assert history(client, a)["items"][0]["id"] == made["id"]  # now the newest
