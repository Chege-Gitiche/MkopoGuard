"""Step 4.3 on the shipped champion: every behaviour check, on the bundle's golden examples.

Runs in CI (the bundle is in Git). Each check is one test. The known flips are expected
failures (strict): if a new model fixes one, the test "unexpectedly passes" and fails the run,
so KNOWN_FLIPS gets updated on purpose.
"""

import pytest

from mkopoguard import bundle, config
from mkopoguard.explain import behaviour as bh

pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / bundle.MANIFEST).exists(), reason="bundle not packaged yet"
)


@pytest.fixture(scope="module")
def champion():
    return bundle.load_bundle(config.BUNDLE_DIR)


@pytest.fixture(scope="module")
def probe(champion):
    return bh.probe_applicants(bundle.golden_frame(champion.golden, champion.schema))


def _param(check):
    if check.name in bh.KNOWN_FLIPS:
        mark = pytest.mark.xfail(strict=True, reason=bh.KNOWN_FLIPS[check.name])
        return pytest.param(check, id=check.name, marks=mark)
    return pytest.param(check, id=check.name)


@pytest.mark.parametrize("check", [_param(c) for c in bh.CHECKS])
def test_champion_moves_risk_the_expected_way(champion, probe, check):
    result = bh.run_check(champion.predict_proba, probe, check)
    assert result["n"] > 0
    assert result["passed"], (
        f"{check.description}: risk moved the wrong way for {result['share_wrong']:.0%} "
        f"(worst {result['worst_wrong_pts']:.2f} points)"
    )


def test_every_report_runs_on_the_champion(champion, probe):
    result = bh.run_all(champion.predict_proba, probe, bh.REPORTS)
    assert (result["n"] > 0).all() and result["passed"].isna().all()
