"""Step 7.2: the traceability matrix stays true, and the checker catches what it should."""

import pytest

from mkopoguard.traceability import MATRIX_HEADING, PLAN, check, check_plan

# --- the real plan ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    return check_plan()


def test_every_test_the_plan_names_exists(report):
    assert not report.broken, report.broken


def test_every_requirement_is_in_the_matrix_once_defined(report):
    assert report.requirements, f"no requirements found in {PLAN}"
    assert not report.unlisted, report.unlisted
    assert not report.unknown, report.unknown


def test_every_requirement_has_a_test_or_a_planned_one(report):
    assert not report.untested, report.untested


def test_every_gap_says_which_step_closes_it(report):
    for req, note in report.gaps.items():
        assert "step 7." in note, f"{req}: {note}"


def test_requirement_ids_are_numbered_in_order(report):
    for prefix in ("FR", "NFR"):
        numbers = [int(r.split("-")[1]) for r in report.requirements if r.startswith(prefix + "-")]
        assert numbers == list(range(1, len(numbers) + 1)), prefix


# --- the checker, on small made-up plans ------------------------------------------------------


@pytest.fixture
def tests_dir(tmp_path):
    folder = tmp_path / "tests" / "unit"
    folder.mkdir(parents=True)
    (folder / "test_x.py").write_text(
        "def test_one():\n    pass\n\nclass TestGroup:\n    def test_two(self):\n        pass\n",
        encoding="utf-8",
    )
    return tmp_path


def plan(requirements, rows):
    lines = ["| ID | Requirement | Priority |", "|---|---|---|"]
    lines += [f"| {req} | Something | Must |" for req in requirements]
    lines += ["", MATRIX_HEADING, "", "| Requirement | Tests | Status |", "|---|---|---|"]
    lines += [f"| {req} | {tests} | {status} |" for req, tests, status in rows]
    return "\n".join(lines)


def test_a_complete_plan_passes(tests_dir):
    text = plan(
        ["FR-01", "FR-02"],
        [
            ("FR-01", "`tests/unit/test_x.py::test_one`", "Covered"),
            ("FR-02", "`tests/unit/test_x.py::test_two`", "Covered"),
        ],
    )
    result = check(text, root=tests_dir)
    assert result.ok(strict=True) and result.covered == ["FR-01", "FR-02"]


def test_a_renamed_or_missing_test_is_caught(tests_dir):
    text = plan(
        ["FR-01"],
        [("FR-01", "`tests/unit/test_x.py::test_gone`<br>`tests/unit/test_nope.py::test_one`", "")],
    )
    result = check(text, root=tests_dir)
    assert len(result.broken) == 2 and not result.ok()


def test_requirements_missing_from_or_unknown_to_the_matrix_are_caught(tests_dir):
    one = "`tests/unit/test_x.py::test_one`"
    text = plan(["FR-01", "FR-02"], [("FR-01", one, ""), ("FR-09", one, "")])
    result = check(text, root=tests_dir)
    assert result.unlisted == ["FR-02"] and result.unknown == ["FR-09"]


def test_a_requirement_without_tests_fails_unless_it_is_a_planned_gap(tests_dir):
    untested = check(plan(["FR-01"], [("FR-01", "", "Covered")]), root=tests_dir)
    assert untested.untested == ["FR-01"] and not untested.ok()
    planned = check(plan(["FR-01"], [("FR-01", "", "Gap: step 7.4")]), root=tests_dir)
    assert planned.ok() and not planned.ok(strict=True)


def test_a_plan_without_a_matrix_is_refused():
    with pytest.raises(ValueError, match="Traceability"):
        check("| FR-01 | Something | Must |")
