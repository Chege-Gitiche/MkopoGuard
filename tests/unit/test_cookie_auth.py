"""Step 6.3: the web app's session cookie, the X-Requested-With guard and the CSP header."""

from types import SimpleNamespace

import pytest
from fastapi import Response
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api import db, security
from mkopoguard.api.auth import LOGIN_FAILED, NOT_FROM_APP, SESSION_COOKIE, set_session_cookie
from mkopoguard.api.hardening import (
    CONTENT_SECURITY_POLICY,
    DOCS_CONTENT_SECURITY_POLICY,
    content_security_policy,
)
from mkopoguard.api.main import create_app
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import create_user, set_active

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
EMAIL = "officer@bank.example"
FROM_APP = {"X-Requested-With": "fetch"}

pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / "manifest.json").exists(), reason="bundle not packaged yet"
)


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(security, "BCRYPT_ROUNDS", 4)
    monkeypatch.setattr(security, "_DUMMY_HASH", None)


@pytest.fixture
def url(tmp_path):
    url = f"sqlite:///{(tmp_path / 'cookie.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        create_user(session, EMAIL, "Officer", PASSWORD, "officer")
    engine.dispose()
    return url


@pytest.fixture
def client(url):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


def app_login(client, password=PASSWORD):
    return client.post(
        "/auth/login", data={"username": EMAIL, "password": password}, headers=FROM_APP
    )


# --- logging in from the app ------------------------------------------------------------------


def test_app_login_sets_an_httponly_strict_cookie_and_returns_no_token(client):
    response = app_login(client)
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] is None and body["token_type"] == "cookie"
    assert body["expires_in"] == 30 * 60
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    lowered = cookie.lower()
    assert "httponly" in lowered
    assert "samesite=strict" in lowered
    assert "path=/" in lowered
    assert "max-age=1800" in lowered


def test_login_without_the_app_header_still_returns_a_bearer_token(client):
    """/docs, scripts and the smoke test keep working exactly as before."""
    response = client.post("/auth/login", data={"username": EMAIL, "password": PASSWORD})
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer" and response.json()["access_token"]
    assert "set-cookie" not in response.headers


def test_a_failed_app_login_sets_no_cookie(client):
    response = app_login(client, password="wrong password entirely")
    assert response.status_code == 401
    assert response.json()["detail"] == LOGIN_FAILED
    assert "set-cookie" not in response.headers


def test_the_cookie_logs_the_officer_in(client):
    app_login(client)
    assert client.get("/auth/me").json()["email"] == EMAIL  # cookie sent automatically


def test_no_cookie_and_no_header_is_401(client):
    assert client.get("/auth/me").status_code == 401


def test_a_tampered_cookie_is_401(client):
    client.cookies.set(SESSION_COOKIE, "not-a-real-token")
    assert client.get("/auth/me").status_code == 401


def test_a_deactivated_officers_cookie_stops_working(client, url):
    app_login(client)
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        set_active(session, EMAIL, False)
    engine.dispose()
    assert client.get("/auth/me").status_code == 401


# --- logging out (step 6.4) ----------------------------------------------------------------


def test_logout_clears_the_session_cookie(client):
    app_login(client)
    response = client.post("/auth/logout", headers=FROM_APP)
    assert response.status_code == 204 and response.content == b""
    cookie = response.headers["set-cookie"].lower()
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "max-age=0" in cookie
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    assert client.get("/auth/me").status_code == 401  # the browser no longer sends it


def test_logout_works_with_no_session_or_an_expired_one(client):
    assert client.post("/auth/logout").status_code == 204
    client.cookies.set(SESSION_COOKIE, "an-expired-or-broken-token")
    assert client.post("/auth/logout").status_code == 204


def test_logout_is_never_cached(client):
    assert client.post("/auth/logout").headers["cache-control"] == "no-store"


# --- the cross-site request forgery guard ------------------------------------------------------


def test_a_change_with_the_cookie_but_without_the_app_header_is_refused(client):
    app_login(client)
    response = client.post("/predict")  # what a forged form from another site would send
    assert response.status_code == 403
    assert response.json()["detail"] == NOT_FROM_APP


def test_a_change_with_the_cookie_and_the_app_header_gets_past_login(client):
    app_login(client)
    response = client.post("/predict", headers=FROM_APP)
    assert response.status_code == 422  # logged in; only the missing profile is wrong


def test_reading_with_the_cookie_needs_no_app_header(client):
    app_login(client)
    assert client.get("/predictions").status_code == 200


def test_a_bearer_token_never_needs_the_app_header(client):
    token = client.post("/auth/login", data={"username": EMAIL, "password": PASSWORD}).json()
    client.cookies.clear()
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    assert client.post("/predict", headers=headers).status_code == 422


def test_the_header_wins_over_the_cookie(client):
    app_login(client)
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "environment, secure", [("production", True), ("development", False), ("test", False)]
)
def test_the_cookie_is_https_only_on_the_live_site(environment, secure):
    response = Response()
    settings = SimpleNamespace(access_token_minutes=30, environment=environment)
    set_session_cookie(response, "token", settings)
    assert ("secure" in response.headers["set-cookie"].lower()) is secure


# --- Content-Security-Policy -------------------------------------------------------------------


def test_pages_and_api_get_the_strict_policy(client):
    for path in ("/", "/assess.html", "/js/api.js", "/health", "/no-such-page"):
        policy = client.get(path).headers["content-security-policy"]
        assert policy == CONTENT_SECURITY_POLICY, path
    assert "unsafe-inline" not in CONTENT_SECURITY_POLICY
    assert "script-src 'self'" in CONTENT_SECURITY_POLICY
    assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY


def test_the_api_docs_get_their_own_policy(client):
    for path in ("/docs", "/docs/oauth2-redirect", "/redoc"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-security-policy"] == DOCS_CONTENT_SECURITY_POLICY
    assert "https://cdn.jsdelivr.net" in DOCS_CONTENT_SECURITY_POLICY


@pytest.mark.parametrize(
    "path, docs",
    [("/docs", True), ("/docs/oauth2-redirect", True), ("/redoc", True), ("/docsx", False),
     ("/documents", False), ("/assess.html", False)],
)  # fmt: skip
def test_only_the_docs_paths_get_the_looser_policy(path, docs):
    expected = DOCS_CONTENT_SECURITY_POLICY if docs else CONTENT_SECURITY_POLICY
    assert content_security_policy(path) == expected
