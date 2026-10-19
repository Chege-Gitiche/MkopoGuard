"""Steps 7.2 and 7.3: check the traceability matrix and test cases in docs/test_plan.md.

python scripts/check_traceability.py            fails on broken links or untested requirements
python scripts/check_traceability.py --strict   also fails while any planned gap is open
"""

import sys

from mkopoguard.traceability import CASE_TYPES, PLAN, check_plan


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    report = check_plan()
    total = len(report.requirements)
    print(f"{PLAN.name}: {total} requirements, {len(report.covered)} with tests, "
          f"{len(report.gaps)} planned gaps")  # fmt: skip
    cases = report.cases.values()
    kinds = ", ".join(f"{sum(c.kind == k for c in cases)} {k.lower()}" for k in CASE_TYPES)
    automated = sum(bool(c.tests) for c in cases)
    print(f"{len(cases)} test cases ({kinds}); {automated} automated, "
          f"{sum(c.manual for c in cases)} run by hand, "
          f"{sum(c.planned for c in cases)} with automation planned")  # fmt: skip
    for label, items in (
        ("Tests named in the plan that don't exist", report.broken),
        ("Requirements missing from the matrix", report.unlisted),
        ("Matrix rows for requirements that aren't defined", report.unknown),
        ("Requirements with no test and no planned gap", report.untested),
        ("Requirements with no test case", report.without_cases),
        ("Test cases that break the rules", report.bad_cases),
    ):
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  - {item}")
    if report.gaps:
        print("\nPlanned gaps:")
        for req, note in report.gaps.items():
            print(f"  - {req} {report.requirements[req]}: {note}")
    ok = report.ok(strict=strict)
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
