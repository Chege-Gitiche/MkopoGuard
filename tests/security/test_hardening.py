"""Step 5.7: login rate limits, no stack traces or echoed secrets, clean logs, safe headers."""

import logging

import pytest
from fastapi.testclient import TestClient

from mkopoguard.api import db, hardening, security
from mkopoguard.api.hardening import LoginLimiter, redact
from mkopoguard.api.main import create_app
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import create_user

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
WRONG = "not the right password"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- the limiter on its own -----------------------------------------------------------------


def test_five_failures_lock_that_ip_and_email_pair_for_15_minutes():
    clock = Clock()
    limiter = LoginLimiter(clock=clock)
    for _ in range(5):
        assert limiter.check("1.1.1.1", "asha@bank.example") is None
        limiter.failed("1.1.1.1", "asha@bank.example")
    assert limiter.check("1.1.1.1", "ASHA@bank.example ") == 900  # same account, any spelling
    clock.now += 899
    assert limiter.check("1.1.1.1", "asha@bank.example") == 1
    clock.now += 1
    assert limiter.check("1.1.1.1", "asha@bank.example") is None


def test_an_attacker_elsewhere_cannot_lock_the_real_officer_out():
    limiter = LoginLimiter(clock=Clock())
    for _ in range(5):
        limiter.check("6.6.6.6", "asha@bank.example")
        limiter.failed("6.6.6.6", "asha@bank.example")
    assert limiter.check("6.6.6.6", "asha@bank.example") is not None
    assert limiter.check("10.0.0.5", "asha@bank.example") is None  # Asha, from her office


def test_a_successful_login_clears_the_failures():
    limiter = LoginLimiter(clock=Clock())
    for _ in range(4):
        limiter.check("1.1.1.1", "a@b.c")
        limiter.failed("1.1.1.1", "a@b.c")
    limiter.succeeded("1.1.1.1", "a@b.c")
    for _ in range(4):
        limiter.failed("1.1.1.1", "a@b.c")
    assert limiter.check("1.1.1.1", "a@b.c") is None


def test_one_ip_trying_many_emails_is_capped():
    clock = Clock()
    limiter = LoginLimiter(attempts_per_ip=20, ip_window_seconds=300, clock=clock)
    for i in range(20):
        assert limiter.check("6.6.6.6", f"user{i}@bank.example") is None
    assert limiter.check("6.6.6.6", "user99@bank.example") == 300
    assert limiter.check("7.7.7.7", "user99@bank.example") is None
    clock.now += 300
    assert limiter.check("6.6.6.6", "user99@bank.example") is None


def test_memory_stays_bounded_under_a_flood_of_made_up_emails(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(LoginLimiter, "MAX_KEYS", 100)
    limiter = LoginLimiter(attempts_per_ip=10**6, clock=clock)
    for i in range(500):
        limiter.check(f"ip{i}", f"x{i}@spam.example")
        limiter.failed(f"ip{i}", f"x{i}@spam.example")
    clock.now += 10_000  # everything expired
    limiter.check("fresh", "fresh@bank.example")
    assert len(limiter._attempts) <= 101 and len(limiter._failures) <= 501
    limiter.failed("fresh", "fresh@bank.example")
    limiter.check("fresh2", "fresh2@bank.example")
    assert len(limiter._failures) <= 101


# --- the limiter in the API -------------------------------------------------------------------


@pytest.fixture
def url(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "BCRYPT_ROUNDS", 4)
    url = f"sqlite:///{(tmp_path / 'hard.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        create_user(session, "asha@bank.example", "Asha", PASSWORD)
    engine.dispose()
    return url


def make_client(url, **settings) -> TestClient:
    values = {"secret_key": SECRET, "database_url": url} | settings
    return TestClient(create_app(Settings(_env_file=None, **values)), raise_server_exceptions=False)


def login(client, email="asha@bank.example", password=PASSWORD):
    return client.post("/auth/login", data={"username": email, "password": password})


def test_the_sixth_wrong_password_is_refused_even_if_the_seventh_is_right(url):
    with make_client(url) as client:
        for _ in range(5):
            assert login(client, password=WRONG).status_code == 401
        blocked = login(client)  # the right password, but too late
        assert blocked.status_code == 429
        assert 0 < int(blocked.headers["Retry-After"]) <= 900
        assert "Too many login attempts" in blocked.json()["detail"]


def test_being_rate_limited_does_not_reveal_whether_an_account_exists(url):
    with make_client(url) as client:
        for email in ("asha@bank.example", "nobody@bank.example"):
            for _ in range(5):
                login(client, email, WRONG)
        real, fake = login(client, "asha@bank.example"), login(client, "nobody@bank.example")
        assert real.status_code == fake.status_code == 429
        assert real.json() == fake.json()


def test_a_correct_login_before_the_limit_resets_the_count(url):
    with make_client(url) as client:
        for _ in range(4):
            login(client, password=WRONG)
        assert login(client).status_code == 200
        for _ in range(4):
            assert login(client, password=WRONG).status_code == 401
        assert login(client).status_code == 200


def test_the_ip_cap_applies_across_emails(url):
    with make_client(url, login_attempts_per_ip=3) as client:
        for i in range(3):
            login(client, f"guess{i}@bank.example", WRONG)
        assert login(client).status_code == 429


# --- no stack traces, no echoed secrets ---------------------------------------------------------


def test_an_unexpected_error_is_a_plain_500_with_a_reference(url, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise RuntimeError("database password is hunter2 at /srv/app/secret_file.py")

    monkeypatch.setattr("mkopoguard.api.auth.find_user", explode)
    with make_client(url) as client, caplog.at_level(logging.ERROR):
        response = login(client)
    assert response.status_code == 500
    body = response.text
    for leak in ("hunter2", "secret_file.py", "Traceback", "RuntimeError"):
        assert leak not in body
    reference = response.json()["reference"]
    assert reference == response.headers["X-Request-ID"]
    assert "Traceback" in caplog.text and "RuntimeError" in caplog.text  # all of it, server-side
    assert any(r.request_id == reference for r in caplog.records if r.levelno == logging.ERROR)


def test_a_validation_error_never_echoes_what_was_sent(url):
    with make_client(url) as client:
        response = client.post(
            "/auth/login",
            data={"username": "asha@bank.example", "password": "MySecretPass!!", "grant_type": "x"},
        )
    assert response.status_code == 422
    assert "MySecretPass" not in response.text
    assert set(response.json()["detail"][0]) == {"loc", "msg", "type"}


def test_logs_never_contain_the_password_the_token_or_the_email(url, caplog):
    with make_client(url) as client, caplog.at_level(logging.DEBUG):
        token = login(client).json()["access_token"]
        client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        login(client, password=WRONG)
    assert caplog.records  # something was logged
    for secret in (PASSWORD, WRONG, token, "asha@bank.example"):
        assert secret not in caplog.text


def test_failed_logins_are_logged_with_a_tag_that_links_attempts(url, caplog):
    with make_client(url) as client, caplog.at_level(logging.INFO):
        login(client, password=WRONG)
        login(client, password=WRONG)
    failures = [r.getMessage() for r in caplog.records if "Login failed" in r.getMessage()]
    tag = hardening.email_tag("asha@bank.example")
    assert len(failures) == 2 and all(tag in line for line in failures)


@pytest.mark.parametrize(
    "line, hidden",
    [
        ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
        ("token eyJhbGciOi.eyJzdWIiOiI3In0.c2lnbmF0dXJl here", "eyJhbGciOi"),
        ("form username=a@b.c&password=hunter2&x=1", "hunter2"),
        ('{"password": "hunter2"}', "hunter2"),
    ],
)
def test_the_log_filter_masks_secrets_that_slip_through(line, hidden):
    assert hidden not in redact(line)


# --- headers --------------------------------------------------------------------------------------


def test_every_response_has_a_request_id_and_security_headers(url):
    with make_client(url) as client:
        for path in ("/health", "/", "/docs", "/no-such-page"):
            headers = client.get(path).headers
            assert len(headers["X-Request-ID"]) == 16
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert headers["X-Frame-Options"] == "DENY"
            assert headers["Referrer-Policy"] == "no-referrer"


def test_api_responses_with_personal_data_are_never_cached(url):
    with make_client(url) as client:
        assert login(client).headers["Cache-Control"] == "no-store"
        assert client.get("/predictions").headers["Cache-Control"] == "no-store"  # even a 401
        assert "no-store" not in client.get("/health").headers.get("Cache-Control", "")


def test_each_request_gets_its_own_id(url):
    with make_client(url) as client:
        ids = {client.get("/health").headers["X-Request-ID"] for _ in range(5)}
    assert len(ids) == 5
