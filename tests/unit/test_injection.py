"""Step 7.3: injection attempts (test cases in docs/test_plan.md, requirement NFR-19).

Whatever an attacker types reaches the database only as a value, never as SQL, and reaches the
page only as text, never as HTML. Every attempt gets an ordinary answer (401, 422, or an empty
list), never a 500.
"""

import json

import pytest
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api import db, security
from mkopoguard.api.auth import LOGIN_FAILED
from mkopoguard.api.main import create_app
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import create_user

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
EMAIL = "officer@bank.example"
TARGET = "target@bank.example"  # attacked with SQL passwords (its own rate-limit counter)
SAMPLE = config.PROJECT_ROOT / "samples" / "profiles" / "sample_10_thin_file.json"
SQL = [
    "' OR '1'='1",
    "officer@bank.example' --",
    "'; DROP TABLE users; --",
    '" OR ""="',
    "%' OR 1=1 --",
]

pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / "manifest.json").exists() or not SAMPLE.exists(),
    reason="bundle or samples not made yet",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    url = f"sqlite:///{(tmp_path_factory.mktemp('inject') / 'i.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    security.BCRYPT_ROUNDS, rounds = 4, security.BCRYPT_ROUNDS
    with db.make_sessions(engine)() as session:
        create_user(session, EMAIL, "Officer", PASSWORD)
        create_user(session, TARGET, "Target", PASSWORD)
    security.BCRYPT_ROUNDS = rounds
    engine.dispose()
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    token = client.post("/auth/login", data={"username": EMAIL, "password": PASSWORD})
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


@pytest.mark.parametrize("attempt", SQL)
def test_sql_in_the_email_is_just_a_wrong_login(client, attempt):
    response = client.post("/auth/login", data={"username": attempt, "password": "anything at all"})
    assert response.status_code == 401
    assert response.json()["detail"] == LOGIN_FAILED


@pytest.mark.parametrize("attempt", SQL)
def test_sql_in_the_password_is_just_a_wrong_login(client, attempt):
    response = client.post("/auth/login", data={"username": TARGET, "password": attempt})
    assert response.status_code == 401


@pytest.mark.parametrize("attempt", SQL)
def test_sql_in_the_history_filter_finds_nothing(client, auth, attempt):
    response = client.get("/predictions", headers=auth, params={"ref": attempt})
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_the_users_table_survives(client):
    """After every attempt above, the real officer can still sign in."""
    response = client.post("/auth/login", data={"username": EMAIL, "password": PASSWORD})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "reference",
    ["' OR '1'='1", "<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "APP-1; DROP"],
)
def test_markup_or_sql_can_never_be_an_applicant_reference(client, auth, reference):
    profile = json.loads(SAMPLE.read_text(encoding="utf-8")) | {"applicant_ref": reference}
    response = client.post("/predict", headers=auth, data={"profile": json.dumps(profile)})
    assert response.status_code == 422
    assert reference not in response.text  # the refused value is never echoed back
