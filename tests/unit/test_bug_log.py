"""Step 7.6: the bug log stays true, and its checker catches what it should."""

import pytest

from mkopoguard.bug_log import LOG, check, check_log

# --- the real log -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def log():
    return check_log()


def test_every_bug_is_well_formed_and_names_a_regression_test_that_exists(log):
    assert log.bugs, f"no bugs found in {LOG}"
    assert not log.problems, log.problems


def test_no_critical_or_major_bug_is_open(log):
    """Exit criterion (docs/test_strategy.md, section 5): no open S1 or S2 defects."""
    blocking = [b for b, bug in log.open.items() if bug.severity.startswith(("S1", "S2"))]
    assert not blocking, blocking


def test_every_open_bug_is_a_known_risk_in_the_strategy(log):
    strategy = (LOG.parent / "test_strategy.md").read_text(encoding="utf-8")
    for bug_id in log.open:
        assert bug_id in strategy, f"{bug_id} is open but not in the strategy's known risks"


# --- the checker, on small made-up logs ------------------------------------------------------

HEADER = (
    "| ID | Date found | Summary | Severity | Steps | Root cause | Fix commit | Regression test |\n"
    "|---|---|---|---|---|---|---|---|\n"
)
FIXED = "`tests/unit/test_x.py::test_fixed`"
OPEN = "`tests/unit/test_x.py::test_still_slow`"


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "tests" / "unit"
    folder.mkdir(parents=True)
    (folder / "test_x.py").write_text(
        "import pytest\n\n\ndef test_fixed():\n    pass\n\n\n"
        '@pytest.mark.xfail(strict=True, reason="BUG-002: slow")\n'
        "def test_still_slow():\n    assert False\n\n\n"
        '@pytest.mark.xfail(reason="BUG-002: slow")\n'
        "def test_not_strict():\n    assert False\n",
        encoding="utf-8",
    )
    return tmp_path


def row(bug_id, commit="`abc1234`", tests=FIXED, severity="S3 Minor", found="2026-10-19"):
    return f"| {bug_id} | {found} | Something | {severity} | Do it | Why | {commit} | {tests} |\n"


def test_a_fixed_and_an_open_bug_pass(root):
    log = check(HEADER + row("BUG-001") + row("BUG-002", "Open", OPEN), root=root)
    assert log.problems == [] and list(log.open) == ["BUG-002"]


@pytest.mark.parametrize(
    "bad_row, problem",
    [
        (row("BUG-001", tests="no test yet"), "names no regression test"),
        (row("BUG-001", tests="`tests/unit/test_x.py::test_gone`"), "doesn't exist"),
        (row("BUG-001", severity="High"), "severity must be"),
        (row("BUG-001", commit="fixed it"), "fix commit must be"),
        (row("BUG-001", found="19/10/2026"), "date found"),
        (row("BUG-003"), "numbered"),
        ("| BUG-001 | 2026-10-19 | Too few columns |\n", "needs 8 columns"),
    ],
)
def test_badly_logged_bugs_are_caught(root, bad_row, problem):
    log = check(HEADER + bad_row, root=root)
    assert any(problem in p for p in log.problems), log.problems


@pytest.mark.parametrize(
    "tests", [FIXED, "`tests/unit/test_x.py::test_not_strict`"], ids=["passing", "not strict"]
)
def test_an_open_bug_needs_a_strict_expected_failure_that_names_it(root, tests):
    log = check(HEADER + row("BUG-001") + row("BUG-002", "Open", tests), root=root)
    assert any("strict expected failure" in p for p in log.problems)


def test_an_open_bug_test_must_name_its_own_bug(root):
    log = check(HEADER + row("BUG-001", "Open", OPEN), root=root)  # the test names BUG-002
    assert any("names BUG-001" in p for p in log.problems)
