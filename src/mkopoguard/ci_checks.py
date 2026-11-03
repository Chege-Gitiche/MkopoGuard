"""Step 7.5: checks CI runs on its own test results.

A green run that quietly skipped the model or API tests isn't green. In CI only the data tests
may skip (they need the generated data, which isn't committed); a skip anywhere else means a
tracked file went missing (the model bundle, the samples) and fails the run.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

ALLOWED_TO_SKIP = ("tests.data.",)


@dataclass(frozen=True)
class Skip:
    test: str
    reason: str


def skipped(junit_xml: str) -> list[Skip]:
    """Every skipped test in a pytest JUnit report (expected failures don't count)."""
    root = ET.fromstring(junit_xml)
    found = []
    for case in root.iter("testcase"):
        for child in case:
            if child.tag == "skipped" and child.get("type") != "pytest.xfail":
                test = f"{case.get('classname')}::{case.get('name')}"
                found.append(Skip(test, child.get("message") or ""))
    return found


def unexpected_skips(junit_xml: str, allowed: tuple[str, ...] = ALLOWED_TO_SKIP) -> list[Skip]:
    return [s for s in skipped(junit_xml) if not s.test.startswith(allowed)]
