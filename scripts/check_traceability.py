"""Step 7.2: check the traceability matrix in docs/test_plan.md against the tests.

python scripts/check_traceability.py            fails on broken links or untested requirements
python scripts/check_traceability.py --strict   also fails while any planned gap is open
"""

import sys

from mkopoguard.traceability import PLAN, check_plan


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    report = check_plan()
    total = len(report.requirements)
    print(f"{PLAN.name}: {total} requirements, {len(report.covered)} with tests, "
          f"{len(report.gaps)} planned gaps")  # fmt: skip
    for label, items in (
        ("Tests named in the plan that don't exist", report.broken),
        ("Requirements missing from the matrix", report.unlisted),
        ("Matrix rows for requirements that aren't defined", report.unknown),
        ("Requirements with no test and no planned gap", report.untested),
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
