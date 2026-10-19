"""Step 5.1: the app starts, serves /health and the frontend, and refuses unsafe settings."""

import shutil

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mkopoguard import config
from mkopoguard.api.main import create_app
from mkopoguard.api.settings import ENV_PREFIX, Settings

SECRET = "test-secret-key-that-is-long-enough-1234567890"
pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / "manifest.json").exists(), reason="bundle not packaged yet"
)


def make_settings(**overrides) -> Settings:
    """Settings for tests only: never read the developer's real .env file."""
    values = {"secret_key": SECRET, "environment": "test"} | overrides
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(make_settings())) as c:  # "with" runs startup and shutdown
        yield c


# --- /health -------------------------------------------------------------------------------------


def test_health_is_up_and_names_the_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == config.MODEL_NAME
    assert body["model_version"] == config.MODEL_VERSION
    assert body["uptime_seconds"] >= 0


def test_health_reveals_nothing_secret(client):
    text = client.get("/health").text
    assert SECRET not in text and "sqlite" not in text.lower()


def test_health_needs_no_login(client):
    assert client.get("/health").status_code == 200  # no Authorization header sent


# --- frontend and docs ---------------------------------------------------------------------------


def test_frontend_is_served_from_the_same_origin(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "MkopoGuard" in response.text


def test_api_routes_win_over_frontend_files(tmp_path):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>home</h1>", encoding="utf-8")
    (frontend / "health").write_text("a file, not the API", encoding="utf-8")
    with TestClient(create_app(make_settings(frontend_dir=frontend))) as c:
        assert c.get("/health").json()["status"] == "ok"


def test_unknown_path_is_404(client):
    assert client.get("/no-such-page").status_code == 404


def test_api_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    assert "/health" in client.get("/openapi.json").json()["paths"]


# --- startup refuses a broken model --------------------------------------------------------------


def test_app_refuses_to_start_with_a_corrupted_bundle(tmp_path):
    broken = tmp_path / "champion"
    shutil.copytree(config.BUNDLE_DIR, broken)
    (broken / "model_card.md").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(Exception, match="checksum"):
        with TestClient(create_app(make_settings(bundle_dir=broken))):
            pass


# --- settings ------------------------------------------------------------------------------------


def test_secret_key_is_required():
    with pytest.raises(ValidationError, match="secret_key"):
        Settings(_env_file=None)


def test_short_or_placeholder_secret_is_refused():
    with pytest.raises(ValidationError):
        make_settings(secret_key="short")
    with pytest.raises(ValidationError, match="placeholder"):
        make_settings(secret_key="change-me-to-a-long-random-string-of-at-least-32-characters")


def test_secret_is_hidden_when_settings_are_printed():
    settings = make_settings()
    assert SECRET not in repr(settings) and SECRET not in str(settings.model_dump())


def test_settings_come_from_environment_variables(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}SECRET_KEY", SECRET)
    monkeypatch.setenv(f"{ENV_PREFIX}ACCESS_TOKEN_MINUTES", "15")
    settings = Settings(_env_file=None)
    assert settings.access_token_minutes == 15
    assert settings.secret_key.get_secret_value() == SECRET


def test_settings_come_from_an_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"{ENV_PREFIX}SECRET_KEY={SECRET}\n{ENV_PREFIX}LOG_LEVEL=DEBUG\n")
    assert Settings(_env_file=env).log_level == "DEBUG"


def test_token_lifetime_must_be_short():
    with pytest.raises(ValidationError):
        make_settings(access_token_minutes=24 * 60)


def test_production_refuses_a_local_sqlite_database():
    with pytest.raises(ValidationError, match="Production"):
        make_settings(environment="production")
    ok = make_settings(environment="production", database_url="postgresql://u:p@host:5432/db")
    assert ok.environment == "production"


def test_a_frontend_folder_without_index_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="index.html"):
        make_settings(frontend_dir=tmp_path)


# --- the secrets file is never committed ---------------------------------------------------------


def test_env_file_is_git_ignored_and_the_example_lists_every_setting():
    ignored = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]
    example = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in Settings.model_fields:
        assert f"{ENV_PREFIX}{name.upper()}" in example, name
