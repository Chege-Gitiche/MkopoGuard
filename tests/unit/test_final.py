"""Tests for step 3.12: the rules around opening the test set, and the final report."""

import json

import numpy as np
import pandas as pd
import pytest

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC
from mkopoguard.models import final


def make_table(n=1_500, seed=0):
    """Applicants whose risk depends on savings club membership and age, with loan terms."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    table["term_months"] = rng.choice([1, 3, 6], n)
    table["loan_amount_kes"] = rng.integers(2, 100, n) * 500
    logit = -2.0 + 1.5 * (table["saved_in_savings_club"] == 0) + 0.8 * np.log(table["age"])
    table["true_p"] = 1 / (1 + np.exp(-logit))
    table["defaulted"] = (rng.random(n) < table["true_p"]).astype(int)
    return table


# --- opening the test set -----------------------------------------------------------------


def test_uncommitted_code_is_refused(tmp_path):
    with pytest.raises(PermissionError, match="commit"):
        final.check_can_open("abc1234-dirty", report=tmp_path / "test_report.json")


def test_first_opening_with_committed_code_is_allowed(tmp_path):
    final.check_can_open("abc1234", report=tmp_path / "test_report.json")  # no error


def test_second_opening_needs_a_reason(tmp_path):
    report = tmp_path / "test_report.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="rerun-reason"):
        final.check_can_open("abc1234", report=report)
    final.check_can_open("abc1234", reason="crashed before finishing", report=report)


def test_every_opening_is_logged_with_its_reason(tmp_path):
    log = tmp_path / "test_runs.log"
    final.log_opening("abc1234", None, log)
    final.log_opening("abc1234", "crashed before finishing", log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "first and planned opening" in lines[0]
    assert "commit abc1234" in lines[1] and "crashed before finishing" in lines[1]


def test_frozen_decisions_are_fingerprinted(tmp_path):
    files = {"a": tmp_path / "a.json", "b": tmp_path / "b.json"}
    files["a"].write_text('{"x": 1}', encoding="utf-8")
    files["b"].write_text('{"x": 2}', encoding="utf-8")
    prints = final.frozen_decisions(files)
    assert set(prints) == {"a", "b"} and prints["a"] != prints["b"]
    files["a"].write_text('{"x": 3}', encoding="utf-8")  # a changed decision...
    assert final.frozen_decisions(files)["a"] != prints["a"]  # ...changes its fingerprint


def test_missing_decision_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        final.frozen_decisions({"missing": tmp_path / "nope.json"})


# --- the report -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    table = make_table()
    return final.build_report(table, table["true_p"].to_numpy(), resamples=200), table


def test_report_has_every_section(report):
    result, table = report
    assert result["n"] == len(table)
    assert set(result) >= {"ranking", "calibration", "decision", "groups"}
    assert set(result["ranking"]) == set(final.RANKING)
    assert set(result["groups"]) == {"all", "kenya", "statement", "thin_file"}


def test_intervals_contain_their_estimates(report):
    result, _ = report
    for metric in final.RANKING:
        r = result["ranking"][metric]
        assert r["low"] <= r["value"] <= r["high"]
    for g in result["groups"].values():
        assert g["pr_auc_low"] <= g["pr_auc"] <= g["pr_auc_high"]


def test_true_probabilities_give_a_calibrated_report(report):
    result, _ = report
    assert result["calibration"]["slope"] == pytest.approx(1.0, abs=0.2)
    assert result["calibration"]["mean_predicted"] == pytest.approx(
        result["default_rate"], abs=0.03
    )


def test_group_sizes_add_up(report):
    result, table = report
    groups = result["groups"]
    assert groups["all"]["n"] == len(table)
    assert groups["statement"]["n"] + groups["thin_file"]["n"] == len(table)


def test_report_is_written_as_valid_json(report, tmp_path):
    result, _ = report
    path = final.write_report(result, tmp_path / "test_report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["n"] == result["n"]
    assert loaded["ranking"]["pr_auc"]["value"] == pytest.approx(
        result["ranking"]["pr_auc"]["value"]
    )
