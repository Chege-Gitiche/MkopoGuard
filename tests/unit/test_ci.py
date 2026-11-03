"""Step 7.5: the CI workflow enforces the strategy's gates, and the skip check works.

The workflow file is configuration, but a typo in it silently removes a gate, so it is tested
like code: the gates in docs/test_strategy.md must be the ones CI actually runs.
"""

import tomllib

import pytest

from mkopoguard import config
from mkopoguard.ci_checks import unexpected_skips

WORKFLOW = config.PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
STRATEGY = config.PROJECT_ROOT / "docs" / "test_strategy.md"
yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps(workflow, job):
    return " ".join(str(step.get("run", "")) for step in workflow["jobs"][job]["steps"])


def test_every_push_is_linted_traced_and_tested_with_coverage(workflow):
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare "on" as True
    assert "push" in triggers
    run = steps(workflow, "test")
    for command in ("ruff check .", "ruff format --check .", "check_traceability.py --strict",
                    "pytest", "--cov", "check_skips.py"):  # fmt: skip
        assert command in run, command
    assert "-m e2e" not in run  # browser tests have their own job


def test_the_coverage_gate_is_the_strategys_80_percent():
    pyproject = tomllib.loads((config.PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 80
    assert "coverage of `src/mkopoguard` is 80% or more" in STRATEGY.read_text(encoding="utf-8")


def test_the_coverage_report_is_kept_even_when_tests_fail(workflow):
    job = workflow["jobs"]["test"]
    upload = [s for s in job["steps"] if "upload-artifact" in s.get("uses", "")]
    assert len(upload) == 1 and "htmlcov/" in upload[0]["with"]["path"]
    assert "failure()" in upload[0]["if"]


def test_browser_tests_run_on_pull_requests_to_main_and_on_demand(workflow):
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["pull_request"]["branches"] == ["main"] and "workflow_dispatch" in triggers
    job = workflow["jobs"]["e2e"]
    assert "pull_request" in job["if"] and job["needs"] == "test"
    assert "playwright install --with-deps chromium" in steps(workflow, "e2e")
    assert "pytest -m e2e" in steps(workflow, "e2e")


def test_ci_runs_the_python_used_locally_and_can_only_read_the_repo(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        python = [s for s in job["steps"] if "setup-python" in s.get("uses", "")]
        assert python[0]["with"]["python-version"] == "3.12"


# --- the skip check ----------------------------------------------------------------------


def junit(*cases):
    body = "".join(
        f'<testcase classname="{cls}" name="{name}">'
        + (f'<skipped type="{kind}" message="why"/>' if kind else "")
        + "</testcase>"
        for cls, name, kind in cases
    )
    return f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'


def test_data_tests_may_skip_and_expected_failures_are_not_skips():
    report = junit(
        ("tests.data.test_features_real", "test_one_row_per_applicant", "pytest.skip"),
        ("tests.model.test_behaviour_champion", "test_flip", "pytest.xfail"),
        ("tests.api.test_predict", "test_ok", None),
    )
    assert unexpected_skips(report) == []


@pytest.mark.parametrize("layer", ["unit", "model", "api", "security"])
def test_a_skip_in_any_other_layer_fails_the_run(layer):
    report = junit((f"tests.{layer}.test_x", "test_needs_the_bundle", "pytest.skip"))
    assert [s.test for s in unexpected_skips(report)] == [
        f"tests.{layer}.test_x::test_needs_the_bundle"
    ]
