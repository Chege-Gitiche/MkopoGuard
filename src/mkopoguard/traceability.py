"""Steps 7.2 and 7.3: requirements traceability.

docs/test_plan.md numbers every requirement (FR-01, NFR-03, ...), names the tests that prove
each one in its traceability matrix, and lists the written test cases (TC-01, ...) with the
requirement each covers and how it is run. This module reads the plan and checks it against the
test files themselves:

- every test the plan names really exists (a renamed or deleted test is caught);
- every requirement is in the matrix, and the matrix lists no requirement that isn't defined;
- every requirement has at least one test, or is marked "Gap" with the step that will close it;
- every requirement has at least one test case, and every test case covers a real requirement,
  is numbered in order, has a known type, and is automated, manual or planned.

`python scripts/check_traceability.py` prints the result; `--strict` also fails on gaps.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from mkopoguard import config

PLAN = config.PROJECT_ROOT / "docs" / "test_plan.md"
MATRIX_HEADING = "## 3. Traceability matrix"
CASES_HEADING = "## 4. Test cases"
ROW = re.compile(r"^\|\s*((?:FR|NFR)-\d{2})\s*\|(.*)$")
CASE_ROW = re.compile(r"^\|\s*(TC-\d{2})\s*\|(.*)$")
REQ_ID = re.compile(r"\b(?:FR|NFR)-\d{2}\b")
TEST_REF = re.compile(r"`(tests/[\w/]+\.py)::(\w+)`")
CASE_TYPES = ("Happy", "Boundary", "Negative")


@dataclass
class Case:
    requirements: list[str]
    kind: str
    tests: list[str]
    manual: bool
    planned: bool


@dataclass
class Report:
    requirements: dict[str, str] = field(default_factory=dict)  # id -> requirement text
    links: dict[str, list[str]] = field(default_factory=dict)  # id -> ["file::test", ...]
    gaps: dict[str, str] = field(default_factory=dict)  # id -> what will close the gap
    cases: dict[str, Case] = field(default_factory=dict)  # TC id -> test case
    broken: list[str] = field(default_factory=list)  # tests named in the plan that don't exist
    unlisted: list[str] = field(default_factory=list)  # requirements missing from the matrix
    unknown: list[str] = field(default_factory=list)  # rows naming undefined requirements
    untested: list[str] = field(default_factory=list)  # neither a test nor a planned gap
    without_cases: list[str] = field(default_factory=list)  # requirements with no test case
    bad_cases: list[str] = field(default_factory=list)  # test cases that break the rules

    @property
    def covered(self) -> list[str]:
        return [req for req in self.requirements if self.links.get(req)]

    def ok(self, strict: bool = False) -> bool:
        problems = (
            self.broken
            or self.unlisted
            or self.unknown
            or self.untested
            or self.without_cases
            or self.bad_cases
        )
        planned = any(case.planned for case in self.cases.values())
        return not problems and not (strict and (self.gaps or planned))


def test_names(path: Path) -> set[str]:
    """Names of the test functions in one file (top level and inside test classes)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        is_function = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if is_function and node.name.startswith("test"):
            names.add(node.name)
    return names


def _cells(rest: str) -> list[str]:
    return [cell.strip() for cell in rest.rstrip().rstrip("|").split("|")]


class _Tests:
    """Looks up test names once per file."""

    def __init__(self, root: Path):
        self.root, self.names = root, {}

    def exists(self, file: str, name: str) -> bool:
        if file not in self.names:
            path = self.root / file
            self.names[file] = test_names(path) if path.exists() else None
        return self.names[file] is not None and name in self.names[file]


def check(plan_text: str, root: Path = config.PROJECT_ROOT) -> Report:
    report = Report()
    if MATRIX_HEADING not in plan_text:
        raise ValueError(f"The plan has no '{MATRIX_HEADING}' section")
    defined, rest = plan_text.split(MATRIX_HEADING, 1)
    matrix, _, cases = rest.partition(CASES_HEADING)
    tests = _Tests(root)

    for line in defined.splitlines():
        row = ROW.match(line)
        if row:
            report.requirements[row.group(1)] = _cells(row.group(2))[0]

    seen = set()
    for line in matrix.splitlines():
        row = ROW.match(line)
        if not row:
            continue
        req, cells = row.group(1), _cells(row.group(2))
        seen.add(req)
        if req not in report.requirements:
            report.unknown.append(req)
            continue
        refs = TEST_REF.findall(cells[0] if cells else "")
        report.links[req] = [f"{file}::{name}" for file, name in refs]
        status = cells[1] if len(cells) > 1 else ""
        if status.lower().startswith("gap"):
            report.gaps[req] = status
        report.broken += [f"{req}: {f}::{n}" for f, n in refs if not tests.exists(f, n)]

    for line in cases.splitlines():
        row = CASE_ROW.match(line)
        if not row:
            continue
        case_id, cells = row.group(1), _cells(row.group(2))
        if len(cells) < 6:
            report.bad_cases.append(f"{case_id}: needs requirement, type, steps, data, "
                                    "expected result and how it is run")  # fmt: skip
            continue
        reqs = REQ_ID.findall(cells[0])
        how = cells[5]
        refs = TEST_REF.findall(how)
        case = Case(
            requirements=reqs,
            kind=cells[1],
            tests=[f"{file}::{name}" for file, name in refs],
            manual="manual" in how.lower(),
            planned="planned" in how.lower(),
        )
        report.cases[case_id] = case
        if not reqs:
            report.bad_cases.append(f"{case_id}: names no requirement")
        report.unknown += [f"{case_id}: {r}" for r in reqs if r not in report.requirements]
        if case.kind not in CASE_TYPES:
            report.bad_cases.append(f"{case_id}: type must be one of {', '.join(CASE_TYPES)}")
        if not (refs or case.manual or case.planned):
            report.bad_cases.append(f"{case_id}: say how it runs (a test, Manual or Planned)")
        report.broken += [f"{case_id}: {f}::{n}" for f, n in refs if not tests.exists(f, n)]

    numbers = [int(case_id.split("-")[1]) for case_id in report.cases]
    if numbers != list(range(1, len(numbers) + 1)):
        report.bad_cases.append("test cases must be numbered TC-01, TC-02, ... in order")

    report.unlisted = [req for req in report.requirements if req not in seen]
    report.untested = [
        req
        for req in report.requirements
        if req in seen and not report.links.get(req) and req not in report.gaps
    ]
    with_cases = {req for case in report.cases.values() for req in case.requirements}
    report.without_cases = [req for req in report.requirements if req not in with_cases]
    return report


def check_plan(path: Path = PLAN) -> Report:
    return check(path.read_text(encoding="utf-8"))
