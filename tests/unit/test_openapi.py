"""Step 5.8: the API docs describe every endpoint, and every endpoint works end to end."""

import importlib.util
import json

import pytest
from fastapi.testclient import TestClient

from mkopoguard import config
from mkopoguard.api import db, openapi, security
from mkopoguard.api.main import create_app
from mkopoguard.api.schemas import ApplicantProfile
from mkopoguard.api.settings import Settings
from mkopoguard.api.users import create_user

SECRET = "test-secret-key-that-is-long-enough-1234567890"
PASSWORD = "correct horse battery staple"
SAMPLES = config.PROJECT_ROOT / "samples"
PUBLIC = {("get", "/health"), ("post", "/auth/login"), ("post", "/auth/logout")}
pytestmark = pytest.mark.skipif(
    not (SAMPLES / "manifest.json").exists(), reason="samples not made yet (step 5.4)"
)


@pytest.fixture(scope="module")
def url(tmp_path_factory):
    url = f"sqlite:///{(tmp_path_factory.mktemp('docs') / 'd.db').as_posix()}"
    db.upgrade(url)
    engine = db.make_engine(url)
    security.BCRYPT_ROUNDS, rounds = 4, security.BCRYPT_ROUNDS
    with db.make_sessions(engine)() as session:
        create_user(session, "officer@bank.example", "Officer", PASSWORD)
        create_user(session, "demo@bank.example", "Demo", PASSWORD, "demo")
    security.BCRYPT_ROUNDS = rounds
    engine.dispose()
    return url


@pytest.fixture(scope="module")
def client(url):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def spec(client):
    return client.get("/openapi.json").json()


def operations(spec):
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            yield method, path, operation


# --- the docs --------------------------------------------------------------------------------


def test_the_docs_page_loads(client):
    page = client.get("/docs")
    assert page.status_code == 200 and "swagger" in page.text.lower()


def test_every_endpoint_has_a_summary_one_group_and_a_description(spec):
    for method, path, op in operations(spec):
        assert op.get("summary"), (method, path)
        assert len(op.get("tags", [])) == 1, (method, path, op.get("tags"))
        assert op.get("description"), (method, path)  # from the function's docstring


def test_every_group_is_described(spec):
    used = {op["tags"][0] for _, _, op in operations(spec)}
    described = {tag["name"] for tag in spec.get("tags", []) if tag.get("description")}
    assert used <= described


def test_every_protected_endpoint_says_it_needs_a_login_and_documents_401(spec):
    for method, path, op in operations(spec):
        if (method, path) in PUBLIC:
            assert "security" not in op, (method, path)
        else:
            assert op.get("security"), (method, path)
            assert "401" in op["responses"], (method, path)


def test_every_endpoint_with_input_documents_its_422(spec):
    for method, path, op in operations(spec):
        if op.get("parameters") or op.get("requestBody"):
            assert "422" in op["responses"], (method, path)


def test_the_login_documents_rate_limiting(spec):
    assert "429" in spec["paths"]["/auth/login"]["post"]["responses"]


def test_the_docs_example_is_a_valid_profile_and_a_real_sample():
    profile = ApplicantProfile.model_validate_json(openapi.EXAMPLE_PROFILE_JSON)
    sample = SAMPLES / "profiles" / "sample_10_thin_file.json"
    if sample.exists():
        assert json.loads(sample.read_text()) == openapi.EXAMPLE_PROFILE
    assert profile.applicant_ref == "SAMPLE-10"


def test_the_predict_form_is_prefilled_with_the_example(spec):
    schema_name = spec["paths"]["/predict"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"].split("/")[-1]
    profile = spec["components"]["schemas"][schema_name]["properties"]["profile"]
    assert profile["example"] == openapi.EXAMPLE_PROFILE_JSON


def test_the_docs_warn_it_is_not_for_real_lending(spec):
    assert "not for real lending decisions" in spec["info"]["description"].lower()


def test_the_docs_example_scores_in_the_api(client):
    token = client.post(
        "/auth/login", data={"username": "officer@bank.example", "password": PASSWORD}
    ).json()["access_token"]
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {token}"},
        data={"profile": openapi.EXAMPLE_PROFILE_JSON},
    )
    assert response.status_code == 200 and response.json()["has_statement"] is False


# --- scripts/api_smoke.py: every endpoint once, for an officer and for the demo account -------


@pytest.fixture(scope="module")
def smoke():
    path = config.PROJECT_ROOT / "scripts" / "api_smoke.py"
    spec = importlib.util.spec_from_file_location("api_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("email", ["officer@bank.example", "demo@bank.example"])
def test_the_smoke_test_passes_end_to_end(client, smoke, email):
    checks = smoke.run(client, email, PASSWORD)
    failed = [c.name for c in checks if not c.passed]
    assert not failed and len(checks) >= 18
