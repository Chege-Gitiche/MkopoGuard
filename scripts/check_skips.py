"""Step 7.5: fail the CI run if any test other than a data test was skipped.

python scripts/check_skips.py test-results/junit.xml
"""

import sys
from pathlib import Path

from mkopoguard.ci_checks import ALLOWED_TO_SKIP, skipped, unexpected_skips


def main() -> int:
    report = Path(sys.argv[1] if len(sys.argv) > 1 else "test-results/junit.xml")
    text = report.read_text(encoding="utf-8")
    everything, bad = skipped(text), unexpected_skips(text)
    print(f"{len(everything)} skipped; allowed only under {', '.join(ALLOWED_TO_SKIP)}")
    for skip in bad:
        print(f"  - {skip.test}: {skip.reason}")
    print("OK" if not bad else f"FAILED: {len(bad)} test(s) skipped that must run in CI")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
