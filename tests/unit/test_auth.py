"""Step 5.3: passwords, tokens, login and the guard on officer routes."""

import importlib.util
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api import db, security
from mkopoguard.api.auth import LOGIN_FAILED, require_roles
from mkopoguard.api.main import create_app
from mkopoguard.api.models import User
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import UserExists, create_user, find_user, set_active

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    """bcrypt at 4 rounds instead of 12: the same algorithm, ~200x faster, for tests only."""
    monkeypatch.setattr(security, "BCRYPT_ROUNDS", 4)
    monkeypatch.setattr(security, "_DUMMY_HASH", None)


@pytest.fixture
def url(tmp_path):
    url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    db.upgrade(url)
    return url


@pytest.fixture
def session(url):
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as s:
        yield s
    engine.dispose()


@pytest.fixture
def accounts(session):
    made = {
        role: create_user(session, f"{role}@bank.example", role.title(), PASSWORD, role)
        for role in ("officer", "admin", "demo")
    }
    made["inactive"] = create_user(session, "gone@bank.example", "Gone", PASSWORD)
    set_active(session, "gone@bank.example", False)
    return made


@pytest.fixture
def client(url, accounts):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


def login(client, email, password=PASSWORD):
    return client.post("/auth/login", data={"username": email, "password": password})


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- passwords ---------------------------------------------------------------------------------


def test_a_password_is_stored_only_as_a_salted_hash():
    first, second = security.hash_password(PASSWORD), security.hash_password(PASSWORD)
    assert PASSWORD not in first and first.startswith("$2b$")
    assert first != second  # a new salt every time
    assert security.verify_password(PASSWORD, first) and security.verify_password(PASSWORD, second)


@pytest.mark.parametrize("wrong", ["Correct horse battery staple", "", PASSWORD + " "])
def test_a_wrong_password_does_not_verify(wrong):
    assert not security.verify_password(wrong, security.hash_password(PASSWORD))


def test_verify_never_raises_on_garbage():
    assert not security.verify_password(PASSWORD, "not-a-bcrypt-hash")
    assert not security.verify_password("x" * 100, security.hash_password(PASSWORD))


@pytest.mark.parametrize(
    "password, message",
    [
        ("short-pass", "at least 12"),
        ("é" * 40, "at most 72 bytes"),  # 40 characters, but 80 bytes
        ("Password123456", "too common"),
        ("asha@bank.example", "email"),
    ],
)
def test_weak_passwords_are_refused(password, message):
    with pytest.raises(security.WeakPassword, match=message):
        security.check_password_policy(password, "asha@bank.example")


def test_a_long_passphrase_without_symbols_is_fine():
    security.check_password_policy("correct horse battery staple", "asha@bank.example")


# --- tokens -------------------------------------------------------------------------------------


def test_a_token_carries_only_the_user_id_and_role():
    claims = security.decode_access_token(
        security.create_access_token(7, "officer", SECRET, 30), SECRET
    )
    assert claims["sub"] == "7" and claims["role"] == "officer"
    assert claims["exp"] - claims["iat"] == 30 * 60
    assert not {"email", "password", "hashed_password"} & set(claims)


def test_an_expired_token_is_refused():
    old = datetime.now(UTC) - timedelta(minutes=31)
    token = security.create_access_token(7, "officer", SECRET, 30, now=old)
    with pytest.raises(security.InvalidToken, match="expired"):
        security.decode_access_token(token, SECRET)


def test_a_token_signed_with_another_key_is_refused():
    token = security.create_access_token(
        7, "officer", "another-secret-key-that-is-long-enough!!", 30
    )
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(token, SECRET)


def test_an_unsigned_token_is_refused():
    now = datetime.now(UTC)
    claims = {
        "sub": "7",
        "role": "admin",
        "iss": security.ISSUER,
        "aud": security.AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(jwt.encode(claims, None, algorithm="none"), SECRET)


def test_a_tampered_token_is_refused():
    token = security.create_access_token(7, "officer", SECRET, 30)
    header, payload, signature = token.split(".")
    forged = jwt.encode({"sub": "1", "role": "admin"}, "x" * 32).split(".")[1]
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(f"{header}.{forged}.{signature}", SECRET)


# --- POST /auth/login ---------------------------------------------------------------------------


def test_login_returns_a_short_lived_bearer_token(client):
    response = login(client, "officer@bank.example")
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 30 * 60
    assert security.decode_access_token(body["access_token"], SECRET)["role"] == "officer"


def test_login_ignores_email_case_and_spaces(client):
    assert login(client, "  OFFICER@Bank.Example ").status_code == 200


@pytest.mark.parametrize(
    "email, password",
    [
        ("officer@bank.example", "wrong password here"),  # wrong password
        ("nobody@bank.example", PASSWORD),  # no such account
        ("gone@bank.example", PASSWORD),  # deactivated
    ],
)
def test_every_failed_login_gets_the_same_answer(client, email, password):
    response = login(client, email, password)
    assert response.status_code == 401
    assert response.json() == {"detail": LOGIN_FAILED}
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_unknown_email_still_checks_a_password(client, monkeypatch):
    """Same work either way, so response time doesn't reveal which emails have accounts."""
    calls = []
    real = security.verify_password
    monkeypatch.setattr(security, "verify_password", lambda p, h: calls.append(h) or real(p, h))
    login(client, "nobody@bank.example")
    assert calls == [security.dummy_hash()]


def test_login_without_a_password_is_a_422_not_a_500(client):
    response = client.post("/auth/login", data={"username": "officer@bank.example"})
    assert response.status_code == 422


# --- GET /auth/me and the guard on officer routes ------------------------------------------------


def test_me_shows_the_logged_in_user_without_the_password_hash(client):
    token = login(client, "officer@bank.example").json()["access_token"]
    body = client.get("/auth/me", headers=bearer(token)).json()
    assert body["email"] == "officer@bank.example" and body["role"] == "officer"
    assert "hashed_password" not in body and "password" not in str(body).lower()


@pytest.mark.parametrize(
    "headers",
    [{}, bearer("not-a-token"), {"Authorization": "Basic b2ZmaWNlcjpwdw=="}],
    ids=["no token", "garbage token", "wrong scheme"],
)
def test_me_needs_a_valid_token(client, headers):
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_expired_token_is_refused_by_the_api(client, accounts):
    old = datetime.now(UTC) - timedelta(hours=1)
    token = security.create_access_token(accounts["officer"].id, "officer", SECRET, 30, now=old)
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_deactivating_an_account_ends_its_session_at_once(client, url):
    token = login(client, "officer@bank.example").json()["access_token"]
    assert client.get("/auth/me", headers=bearer(token)).status_code == 200
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        set_active(session, "officer@bank.example", False)
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_a_token_for_a_user_that_does_not_exist_is_refused(client):
    token = security.create_access_token(9999, "admin", SECRET, 30)
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_roles_are_enforced(accounts):
    only_admins = require_roles("admin")
    assert only_admins(accounts["admin"]) is accounts["admin"]
    for role in ("officer", "demo"):
        with pytest.raises(HTTPException) as error:
            only_admins(accounts[role])
        assert error.value.status_code == 403


def test_login_and_me_are_in_the_api_docs(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/auth/login" in paths and "/auth/me" in paths


# --- accounts -----------------------------------------------------------------------------------


def test_creating_an_account_stores_a_hash_and_normalises_the_email(session):
    user = create_user(session, " Wanjiku@Bank.Example", "Wanjiku M", PASSWORD)
    assert user.email == "wanjiku@bank.example" and user.role == "officer"
    assert user.hashed_password != PASSWORD
    assert security.verify_password(PASSWORD, user.hashed_password)


def test_one_account_per_email(session):
    create_user(session, "wanjiku@bank.example", "Wanjiku M", PASSWORD)
    with pytest.raises(UserExists):
        create_user(session, "WANJIKU@bank.example", "Someone Else", PASSWORD)


def test_accounts_need_a_known_role_a_name_and_a_strong_password(session):
    with pytest.raises(ValueError, match="Role"):
        create_user(session, "a@bank.example", "A", PASSWORD, role="superuser")
    with pytest.raises(ValueError, match="name"):
        create_user(session, "a@bank.example", "  ", PASSWORD)
    with pytest.raises(security.WeakPassword):
        create_user(session, "a@bank.example", "A", "short")
    assert find_user(session, "a@bank.example") is None


def test_the_demo_role_is_allowed_by_the_database(session):
    assert create_user(session, "demo2@bank.example", "Demo", PASSWORD, "demo").role == "demo"


# --- scripts/create_user.py ---------------------------------------------------------------------


@pytest.fixture
def script():
    path = config.PROJECT_ROOT / "scripts" / "create_user.py"
    spec = importlib.util.spec_from_file_location("create_user", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_creates_and_deactivates_accounts(script, url, monkeypatch, session):
    monkeypatch.setenv(script.PASSWORD_ENV, PASSWORD)
    args = ["create", "--email", "Njeri@Bank.Example", "--name", "Njeri K", "--role", "demo"]
    assert script.main(args, url=url) == 0
    assert script.main(["deactivate", "--email", "njeri@bank.example"], url=url) == 0
    user = session.query(User).filter_by(email="njeri@bank.example").one()
    assert user.role == "demo" and user.is_active is False


def test_the_script_refuses_a_weak_password_and_a_duplicate(script, url, monkeypatch, capsys):
    monkeypatch.setenv(script.PASSWORD_ENV, "short")
    with pytest.raises(SystemExit, match="at least 12"):
        script.main(["create", "--email", "x@bank.example", "--name", "X"], url=url)
    monkeypatch.setenv(script.PASSWORD_ENV, PASSWORD)
    assert script.main(["create", "--email", "x@bank.example", "--name", "X"], url=url) == 0
    assert script.main(["create", "--email", "x@bank.example", "--name", "X"], url=url) == 1
    assert "already exists" in capsys.readouterr().err
