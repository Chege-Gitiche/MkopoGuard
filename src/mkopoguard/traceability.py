"""Step 7.2: requirements traceability.

docs/test_plan.md numbers every requirement (FR-01, NFR-03, ...) and, in its traceability
matrix, names the tests that prove each one as `tests/<folder>/<file>.py::<test name>`. This
module reads the plan and checks it against the test files themselves:

- every test the matrix names really exists (a renamed or deleted test is caught);
- every requirement is in the matrix, and the matrix lists no requirement that isn't defined;
- every requirement has at least one test, or is marked "Gap" with the step that will close it.

`python scripts/check_traceability.py` prints the result; `--strict` also fails on gaps.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from mkopoguard import config

PLAN = config.PROJECT_ROOT / "docs" / "test_plan.md"
MATRIX_HEADING = "## 3. Traceability matrix"
ROW = re.compile(r"^\|\s*((?:FR|NFR)-\d{2})\s*\|(.*)$")
TEST_REF = re.compile(r"`(tests/[\w/]+\.py)::(\w+)`")


@dataclass
class Report:
    requirements: dict[str, str] = field(default_factory=dict)  # id -> requirement text
    links: dict[str, list[str]] = field(default_factory=dict)  # id -> ["file::test", ...]
    gaps: dict[str, str] = field(default_factory=dict)  # id -> what will close the gap
    broken: list[str] = field(default_factory=list)  # tests named in the plan that don't exist
    unlisted: list[str] = field(default_factory=list)  # requirements missing from the matrix
    unknown: list[str] = field(default_factory=list)  # matrix rows for undefined requirements
    untested: list[str] = field(default_factory=list)  # neither a test nor a planned gap

    @property
    def covered(self) -> list[str]:
        return [req for req in self.requirements if self.links.get(req)]

    def ok(self, strict: bool = False) -> bool:
        problems = self.broken or self.unlisted or self.unknown or self.untested
        return not problems and not (strict and self.gaps)


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


def check(plan_text: str, root: Path = config.PROJECT_ROOT) -> Report:
    report = Report()
    if MATRIX_HEADING not in plan_text:
        raise ValueError(f"The plan has no '{MATRIX_HEADING}' section")
    defined, matrix = plan_text.split(MATRIX_HEADING, 1)

    for line in defined.splitlines():
        row = ROW.match(line)
        if row:
            report.requirements[row.group(1)] = _cells(row.group(2))[0]

    names_in: dict[str, set[str] | None] = {}
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
        for file, name in refs:
            if file not in names_in:
                path = root / file
                names_in[file] = test_names(path) if path.exists() else None
            if names_in[file] is None or name not in names_in[file]:
                report.broken.append(f"{req}: {file}::{name}")

    report.unlisted = [req for req in report.requirements if req not in seen]
    report.untested = [
        req
        for req in report.requirements
        if req in seen and not report.links.get(req) and req not in report.gaps
    ]
    return report


def check_plan(path: Path = PLAN) -> Report:
    return check(path.read_text(encoding="utf-8"))
