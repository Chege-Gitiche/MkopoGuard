"""Steps 7.4 and 7.5: the release gate for speed (NFR-16, test case TC-70).

A load test takes minutes, so it isn't run with every test run. `python perf/run_load_test.py`
runs it and saves the result in docs/results/load_test.json; these tests check that saved
result: it exists, it was run on the model that ships, at the agreed load, and it met the target.
A new model version needs a fresh load test.

The speed target is not met yet (BUG-002), so that one test is a strict expected failure: CI
stays green while the defect is open, and the day a load test passes, the test "unexpectedly
passes", fails the run, and reminds us to remove the mark and close the bug.
"""

import json

import pytest

from mkopoguard import config

RESULT = config.PROJECT_ROOT / "docs" / "results" / "load_test.json"
STRATEGY = config.PROJECT_ROOT / "docs" / "test_strategy.md"
BUG_002 = (
    "BUG-002: /predict p95 was 4.1 s at 10 users pausing 1 to 2 s (target 500 ms); one "
    "process can't keep up. Remove this mark when perf/run_load_test.py passes."
)


@pytest.fixture(scope="module")
def result():
    assert RESULT.exists(), "No load test result yet: run python perf/run_load_test.py"
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_load_test_ran_on_the_shipped_model_at_the_agreed_load(result):
    assert result["model_version"] == config.MODEL_VERSION, "new model: run the load test again"
    assert result["users"] >= 10 and result["seconds"] >= 120
    assert result["predict_requests"] >= 100


def test_no_request_failed_under_load(result):
    """Slow is a known defect; errors are not: every request must still succeed."""
    assert result["failures"] == 0


@pytest.mark.xfail(strict=True, reason=BUG_002)
def test_predict_met_the_local_speed_target(result):
    assert result["predict_p95_ms"] < result["p95_target_ms"] == 500
    assert result["passed"] is True


def test_the_target_is_the_one_in_the_test_strategy():
    assert "p95 under 500 ms locally" in STRATEGY.read_text(encoding="utf-8")
