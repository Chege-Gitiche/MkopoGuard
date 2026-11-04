"""Step 7.6: the bug log (docs/bug_log.md) is checked like the test plan.

Every row needs an ID in order, the date it was found, a severity from the strategy's scale,
and either the commit that fixed it or "Open". Every bug names its regression test, and the
test must exist. An open bug's test must be a strict expected failure that names the bug, so
the day the bug is fixed the test fails and says "close BUG-nnn".
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from mkopoguard import config
from mkopoguard.traceability import TEST_REF

LOG = config.PROJECT_ROOT / "docs" / "bug_log.md"
ROW = re.compile(r"^\|\s*(BUG-\d{3})\s*\|(.*)$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEVERITY = re.compile(r"^(S[1-4]) (Critical|Major|Minor|Cosmetic)$")
COMMIT = re.compile(r"^`[0-9a-f]{7,40}`$")


@dataclass
class Bug:
    found: str
    summary: str
    severity: str
    fixed_in: str | None  # None while open
    tests: list[str]


@dataclass
class BugLog:
    bugs: dict[str, Bug] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def open(self) -> dict[str, Bug]:
        return {bug_id: bug for bug_id, bug in self.bugs.items() if bug.fixed_in is None}


def _cells(rest: str) -> list[str]:
    return [cell.strip() for cell in rest.rstrip().rstrip("|").split("|")]


def _function(path: Path, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _strict_xfail(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func).endswith("xfail"):
            for keyword in decorator.keywords:
                if keyword.arg == "strict" and getattr(keyword.value, "value", None) is True:
                    return True
    return False


def check(text: str, root: Path = config.PROJECT_ROOT) -> BugLog:
    log = BugLog()
    for line in text.splitlines():
        row = ROW.match(line)
        if not row:
            continue
        bug_id, cells = row.group(1), _cells(row.group(2))
        if len(cells) != 7:
            log.problems.append(f"{bug_id}: needs 8 columns (ID, date found, summary, severity, "
                                "steps, root cause, fix commit, regression test)")  # fmt: skip
            continue
        found, summary, severity, _steps, _cause, commit, tests = cells
        refs = TEST_REF.findall(tests)
        bug = Bug(found, summary, severity, None if commit == "Open" else commit,
                  [f"{f}::{n}" for f, n in refs])  # fmt: skip
        log.bugs[bug_id] = bug
        if not DATE.match(found):
            log.problems.append(f"{bug_id}: date found must look like 2026-10-19")
        if not SEVERITY.match(severity):
            log.problems.append(f"{bug_id}: severity must be S1 Critical, S2 Major, S3 Minor "
                                "or S4 Cosmetic")  # fmt: skip
        if commit != "Open" and not COMMIT.match(commit):
            log.problems.append(f"{bug_id}: fix commit must be a commit hash in backticks, or Open")
        if not refs:
            log.problems.append(f"{bug_id}: names no regression test")
        for file, name in refs:
            path = root / file
            node = _function(path, name) if path.exists() else None
            if node is None:
                log.problems.append(f"{bug_id}: {file}::{name} doesn't exist")
            elif commit == "Open" and not (
                _strict_xfail(node) and bug_id in path.read_text(encoding="utf-8")
            ):
                log.problems.append(f"{bug_id}: open, so {name} must be a strict expected "
                                    f"failure that names {bug_id}")  # fmt: skip
    numbers = [int(bug_id.split("-")[1]) for bug_id in log.bugs]
    if numbers != list(range(1, len(numbers) + 1)):
        log.problems.append("bugs must be numbered BUG-001, BUG-002, ... in order")
    return log


def check_log(path: Path = LOG) -> BugLog:
    return check(path.read_text(encoding="utf-8"))
