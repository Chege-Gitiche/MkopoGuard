"""Steps 7.2 and 7.3: the plan stays true, and the checker catches what it should."""

import pytest

from mkopoguard.traceability import CASES_HEADING, MATRIX_HEADING, PLAN, check, check_plan

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


def test_every_requirement_has_a_written_test_case(report):
    assert not report.without_cases, report.without_cases
    assert not report.bad_cases, report.bad_cases


def test_the_cases_cover_boundaries_and_failures_not_just_happy_paths(report):
    kinds = [case.kind for case in report.cases.values()]
    assert kinds.count("Boundary") >= 10 and kinds.count("Negative") >= 20


def test_the_boundaries_the_guide_names_have_cases(report):
    text = PLAN.read_text(encoding="utf-8")
    cases = text[text.index(CASES_HEADING) :]
    for must_appear in ("Age 17, 18, 100, 101", "1,048,576 bytes", "balance_after",
                        "Wrong password", "Expired token"):  # fmt: skip
        assert must_appear in cases, must_appear


def test_a_case_planned_for_later_still_has_a_requirement_with_a_gap_or_a_test(report):
    for case in report.cases.values():
        if case.planned and not case.tests:
            assert any(req in report.gaps or report.links.get(req) for req in case.requirements)


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


ONE = "`tests/unit/test_x.py::test_one`"


def plan(requirements, rows, cases=None):
    """A small plan; by default one automated happy-path case per requirement."""
    if cases is None:
        cases = [(f"TC-{i:02d}", req, "Happy", ONE) for i, req in enumerate(requirements, 1)]
    lines = ["| ID | Requirement | Priority |", "|---|---|---|"]
    lines += [f"| {req} | Something | Must |" for req in requirements]
    lines += ["", MATRIX_HEADING, "", "| Requirement | Tests | Status |", "|---|---|---|"]
    lines += [f"| {req} | {tests} | {status} |" for req, tests, status in rows]
    lines += ["", CASES_HEADING, "", "| ID | Req | Type | Steps | Data | Expected | How |"]
    lines += [f"| {cid} | {req} | {kind} | Do it | Data | It works | {how} |"
              for cid, req, kind, how in cases]  # fmt: skip
    return "\n".join(lines)


def test_a_complete_plan_passes(tests_dir):
    text = plan(
        ["FR-01", "FR-02"],
        [("FR-01", ONE, "Covered"), ("FR-02", "`tests/unit/test_x.py::test_two`", "Covered")],
    )
    result = check(text, root=tests_dir)
    assert result.ok(strict=True) and result.covered == ["FR-01", "FR-02"]
    assert len(result.cases) == 2


def test_a_renamed_or_missing_test_is_caught(tests_dir):
    text = plan(
        ["FR-01"],
        [("FR-01", "`tests/unit/test_x.py::test_gone`<br>`tests/unit/test_nope.py::test_one`", "")],
    )
    result = check(text, root=tests_dir)
    assert len(result.broken) == 2 and not result.ok()


def test_requirements_missing_from_or_unknown_to_the_matrix_are_caught(tests_dir):
    text = plan(["FR-01", "FR-02"], [("FR-01", ONE, ""), ("FR-09", ONE, "")])
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


# --- test cases ------------------------------------------------------------------------------


def test_a_requirement_without_a_test_case_is_caught(tests_dir):
    text = plan(["FR-01", "FR-02"], [("FR-01", ONE, ""), ("FR-02", ONE, "")],
                cases=[("TC-01", "FR-01", "Happy", ONE)])  # fmt: skip
    assert check(text, root=tests_dir).without_cases == ["FR-02"]


def test_a_case_for_an_unknown_requirement_or_a_missing_test_is_caught(tests_dir):
    cases = [("TC-01", "FR-01", "Happy", ONE),
             ("TC-02", "FR-07", "Happy", "`tests/unit/test_x.py::test_gone`")]  # fmt: skip
    result = check(plan(["FR-01"], [("FR-01", ONE, "")], cases), root=tests_dir)
    assert result.unknown == ["TC-02: FR-07"]
    assert result.broken == ["TC-02: tests/unit/test_x.py::test_gone"]


@pytest.mark.parametrize(
    "cases, problem",
    [
        ([("TC-01", "FR-01", "Sunny", ONE)], "type must be"),
        ([("TC-01", "FR-01", "Happy", "someday")], "say how it runs"),
        ([("TC-02", "FR-01", "Happy", ONE)], "numbered"),
        ([("TC-01", "nothing", "Happy", ONE)], "names no requirement"),
    ],
)
def test_badly_written_cases_are_caught(tests_dir, cases, problem):
    result = check(plan(["FR-01"], [("FR-01", ONE, "")], cases), root=tests_dir)
    assert any(problem in p for p in result.bad_cases), result.bad_cases


def test_manual_and_planned_cases_are_allowed_but_planned_fails_strict(tests_dir):
    cases = [("TC-01", "FR-01", "Happy", "Manual (step 6.5)"),
             ("TC-02", "FR-01", "Negative", "Planned: step 7.4")]  # fmt: skip
    result = check(plan(["FR-01"], [("FR-01", ONE, "")], cases), root=tests_dir)
    assert result.ok() and not result.ok(strict=True)
