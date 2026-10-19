"""Step 7.4: the release gate for speed (NFR-16, test case TC-70).

A load test takes minutes, so it isn't run with every test run. `python perf/run_load_test.py`
runs it and saves the result in docs/results/load_test.json; these tests check that saved
result: it exists, it was run on the model that ships, at the agreed load, and it met the target.
A new model version needs a fresh load test.
"""

import json

import pytest

from mkopoguard import config

RESULT = config.PROJECT_ROOT / "docs" / "results" / "load_test.json"
STRATEGY = config.PROJECT_ROOT / "docs" / "test_strategy.md"


@pytest.fixture(scope="module")
def result():
    assert RESULT.exists(), "No load test result yet: run python perf/run_load_test.py"
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_load_test_ran_on_the_shipped_model_at_the_agreed_load(result):
    assert result["model_version"] == config.MODEL_VERSION, "new model: run the load test again"
    assert result["users"] >= 10 and result["seconds"] >= 120
    assert result["predict_requests"] >= 100


def test_predict_met_the_local_speed_target_with_no_errors(result):
    assert result["failures"] == 0
    assert result["predict_p95_ms"] < result["p95_target_ms"] == 500
    assert result["passed"] is True


def test_the_target_is_the_one_in_the_test_strategy():
    assert "p95 under 500 ms locally" in STRATEGY.read_text(encoding="utf-8")
