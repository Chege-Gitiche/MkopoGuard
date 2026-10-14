"""Step 5.5: POST /predict and GET /model/info, end to end through the real app and bundle."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from mkopoguard import config
from mkopoguard.api import db, security
from mkopoguard.api.main import create_app
from mkopoguard.api.models import Prediction, User
from mkopoguard.api.settings import Settings
from mkopoguard.api.statement import MAX_BYTES
from mkopoguard.api.users import create_user

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
SAMPLES = config.PROJECT_ROOT / "samples"
MANIFEST = json.loads((SAMPLES / "manifest.json").read_text()) if SAMPLES.exists() else []
pytestmark = pytest.mark.skipif(not MANIFEST, reason="samples not made yet (step 5.4)")


@pytest.fixture(scope="module")
def url(tmp_path_factory):
    url = f"sqlite:///{(tmp_path_factory.mktemp('predict') / 'p.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    security.BCRYPT_ROUNDS, rounds = 4, security.BCRYPT_ROUNDS  # fast hashing for tests
    with db.make_sessions(engine)() as session:
        for role in ("officer", "demo"):
            create_user(session, f"{role}@bank.example", role.title(), PASSWORD, role)
        create_user(session, "other@bank.example", "Other Officer", PASSWORD)
    security.BCRYPT_ROUNDS = rounds
    engine.dispose()
    return url


@pytest.fixture(scope="module")
def client(url):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


def token_for(client, email):
    response = client.post("/auth/login", data={"username": email, "password": PASSWORD})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture(scope="module")
def officer(client):
    return token_for(client, "officer@bank.example")


@pytest.fixture(scope="module")
def demo(client):
    return token_for(client, "demo@bank.example")


def sample(name_part: str) -> dict:
    return next(s for s in MANIFEST if name_part in s["name"])


def send(client, headers, profile: str | dict, statement: bytes | None = None, name="s.csv"):
    text = profile if isinstance(profile, str) else json.dumps(profile)
    files = {"statement": (name, statement, "text/csv")} if statement is not None else None
    return client.post("/predict", headers=headers, data={"profile": text}, files=files)


def send_sample(client, headers, s: dict):
    profile = (SAMPLES / s["profile"]).read_text()
    data = (SAMPLES / s["statement"]).read_bytes() if s["statement"] else None
    return send(client, headers, profile, data)


def saved(url, **where) -> list[Prediction]:
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        query = select(Prediction).options(selectinload(Prediction.officer)).filter_by(**where)
        rows = session.scalars(query).all()
    engine.dispose()
    return rows


# --- scoring --------------------------------------------------------------------------------


@pytest.mark.parametrize("s", MANIFEST, ids=[s["name"] for s in MANIFEST])
def test_every_sample_scores_as_recorded_through_the_api(client, officer, s):
    response = send_sample(client, officer, s)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["probability_of_default"] == pytest.approx(s["probability_of_default"], abs=1e-9)
    assert body["decision"] == s["decision"] and body["band"] == s["band"]
    assert body["has_statement"] == (s["statement"] is not None)
    assert len(body["reasons"]["raising"]) == 3 and len(body["reasons"]["lowering"]) == 2
    assert body["model_version"] == config.MODEL_VERSION and body["id"] is not None


def test_a_thin_file_applicant_is_never_given_statement_reasons(client, officer):
    from mkopoguard.explain.reasons import UNITS

    statement_codes = {u.code for u in UNITS if u.needs_statement}
    for s in (x for x in MANIFEST if x["statement"] is None):
        body = send_sample(client, officer, s).json()
        codes = {r["code"] for r in body["reasons"]["raising"] + body["reasons"]["lowering"]}
        assert not codes & statement_codes


def test_the_decision_follows_the_threshold_for_the_term(client, officer):
    body = send_sample(client, officer, sample("medium")).json()
    expected = "decline" if body["probability_of_default"] >= body["threshold"] else "approve"
    assert body["decision"] == expected
    assert body["threshold"] == pytest.approx(
        json.loads((config.BUNDLE_DIR / "decision_policy.json").read_text())[
            "thresholds_by_term_months"
        ][str(body["term_months"])]
    )


def test_an_officers_prediction_is_saved_with_its_model_version(client, officer, url):
    body = send_sample(client, officer, sample("very_high")).json()
    (row,) = saved(url, id=body["id"])
    assert row.model_version == config.MODEL_VERSION
    assert row.probability == pytest.approx(body["probability_of_default"])
    assert row.decision == body["decision"] and row.band == body["band"]
    assert row.reasons["raising"][0]["code"] == body["reasons"]["raising"][0]["code"]
    assert row.officer.email == "officer@bank.example"


def test_saved_inputs_keep_features_but_never_the_raw_transactions(client, officer, url):
    s = sample("low")
    body = send_sample(client, officer, s).json()
    (row,) = saved(url, id=body["id"])
    assert row.inputs["profile"]["applicant_ref"] == body["applicant_ref"]
    assert "stmt_betting_share" in row.inputs["statement_features"]
    assert row.inputs["statement_transactions"] > 0
    first_line = (SAMPLES / s["statement"]).read_text().splitlines()[1]
    assert first_line.split(",")[0] not in json.dumps(row.inputs)  # no timestamps, no rows


def test_the_demo_account_gets_a_result_but_nothing_is_saved(client, demo, url):
    before = len(saved(url))
    response = send_sample(client, demo, sample("high"))
    assert response.status_code == 200
    assert response.json()["id"] is None
    assert len(saved(url)) == before


def test_unusual_inputs_are_scored_with_a_warning(client, officer):
    profile = json.loads((SAMPLES / sample("thin_file")["profile"]).read_text())
    body = send(client, officer, profile | {"age": 100}).json()  # oldest in training: 99
    assert any("age" in w and "outside the training range" in w for w in body["warnings"])


def test_an_empty_file_field_counts_as_no_statement(client, officer):
    """Browsers send an empty, unnamed part when no file was chosen."""
    profile = (SAMPLES / sample("thin_file")["profile"]).read_text()
    response = send(client, officer, profile, b"", name="")
    assert response.status_code == 200 and response.json()["has_statement"] is False


# --- bad input is a clear 422, never a 500 ------------------------------------------------------


def detail(response) -> list[dict]:
    assert response.status_code == 422, response.text
    return response.json()["detail"]


def test_a_profile_that_is_not_json(client, officer):
    assert detail(send(client, officer, "{not json"))[0]["loc"] == ["profile"]


def test_a_bad_profile_field_is_named(client, officer):
    profile = json.loads((SAMPLES / sample("low")["profile"]).read_text())
    found = detail(send(client, officer, profile | {"age": 15, "term_months": 2}))
    assert {tuple(e["loc"]) for e in found} == {("profile", "age"), ("profile", "term_months")}


def test_a_missing_profile_is_named(client, officer):
    found = detail(client.post("/predict", headers=officer, data={}))
    assert any("profile" in e["loc"] for e in found)


def test_a_broken_statement_row_is_named(client, officer):
    s = sample("low")
    lines = (SAMPLES / s["statement"]).read_text().splitlines()
    lines[5] = lines[5].replace(",out,", ",sideways,")
    found = detail(send(client, officer, (SAMPLES / s["profile"]).read_text(),
                        ("\n".join(lines) + "\n").encode()))  # fmt: skip
    assert found[0]["loc"] == ["statement", "row 6"]


def test_a_statement_for_another_period_is_refused(client, officer):
    s = sample("low")
    profile = json.loads((SAMPLES / s["profile"]).read_text()) | {"application_date": "2026-01-01"}
    found = detail(send(client, officer, profile, (SAMPLES / s["statement"]).read_bytes()))
    assert found[0]["loc"] == ["statement", "period"]


def test_a_huge_upload_is_refused_without_reading_it_all(client, officer):
    profile = (SAMPLES / sample("low")["profile"]).read_text()
    found = detail(send(client, officer, profile, b"x" * (MAX_BYTES * 3)))
    assert "1 MB" in found[0]["msg"]


def test_a_file_that_is_not_a_csv(client, officer):
    profile = (SAMPLES / sample("low")["profile"]).read_text()
    found = detail(send(client, officer, profile, b"\x89PNG\r\n\x1a\n\x00\x00", name="photo.png"))
    assert found[0]["loc"] == ["statement", "file"]


def test_nothing_is_saved_when_the_input_is_bad(client, officer, url):
    before = len(saved(url))
    send(client, officer, "{not json")
    assert len(saved(url)) == before


# --- who can score ------------------------------------------------------------------------------


def test_predict_needs_a_login(client):
    profile = (SAMPLES / sample("low")["profile"]).read_text()
    assert send(client, {}, profile).status_code == 401


def test_each_officers_predictions_carry_their_own_id(client, url):
    other = token_for(client, "other@bank.example")
    body = send_sample(client, other, sample("low")).json()
    (row,) = saved(url, id=body["id"])
    assert row.officer.email == "other@bank.example"


# --- GET /model/info ----------------------------------------------------------------------------


def test_model_info_describes_the_model_and_its_rule(client, demo):
    info = client.get("/model/info", headers=demo).json()
    assert info["name"] == config.MODEL_NAME and info["version"] == config.MODEL_VERSION
    assert set(info["thresholds_by_term_months"]) == {"1", "3", "6"}
    assert list(info["bands"]) == ["low", "medium", "high", "very_high"]
    assert 0 < info["test_pr_auc"] < 1 and "not for real lending" in info["notice"]


def test_model_info_needs_a_login(client):
    assert client.get("/model/info").status_code == 401


def test_the_demo_account_never_owns_a_saved_prediction(url):
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        query = select(func.count()).select_from(Prediction).join(Prediction.officer)
        demo_rows = session.scalar(query.where(User.role == "demo"))
    engine.dispose()
    assert demo_rows == 0
