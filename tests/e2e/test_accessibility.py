"""Step 7.4: axe-core on every screen, signed out and signed in (NFR-14, test case TC-68).

Fails on any serious or critical violation. Lighthouse and the keyboard tour are still done by
hand before a release (docs/accessibility.md); axe catches regressions on every run in between.
"""

import re

import pytest
from axe_playwright_python.sync_playwright import Axe
from e2e_helpers import load_sample
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

BLOCKING = {"serious", "critical"}
axe = Axe()


def problems(page: Page) -> list[str]:
    page.wait_for_load_state("networkidle")  # the page has drawn what it fetched
    violations = axe.run(page).response["violations"]
    return [
        f"{v['id']} ({v['impact']}): {v['help']} at {[n['target'] for n in v['nodes']][:3]}"
        for v in violations
        if v["impact"] in BLOCKING
    ]


@pytest.mark.parametrize("path", ["/", "/?reason=signed-out", "/no-such-page"])
def test_signed_out_pages_have_no_serious_violations(page: Page, path):
    page.goto(path)
    assert problems(page) == []


def test_every_signed_in_screen_has_no_serious_violations(officer: Page):
    page = officer
    found = {"assess": problems(page)}

    page.get_by_role("button", name="Assess risk").click()  # the error summary and field errors
    expect(page.locator("#form-errors")).to_be_visible()
    found["assess with errors"] = problems(page)

    load_sample(page, "sample_10_thin_file")
    page.get_by_role("button", name="Assess risk").click()
    expect(page).to_have_url(re.compile(r"/result\.html\?id=\d+$"))
    expect(page.locator("#result")).to_be_visible()
    found["result"] = problems(page)

    page.get_by_role("link", name="History").click()
    expect(page.get_by_role("row").filter(has_text="SAMPLE-10").first).to_be_visible()
    found["history"] = problems(page)

    page.get_by_role("link", name="View SAMPLE-10").first.click()
    expect(page.locator("#detail")).to_be_visible()
    found["detail"] = problems(page)

    page.get_by_role("link", name="Model info").click()
    expect(page.locator("#model-facts")).to_contain_text("1.1.0")
    found["about"] = problems(page)

    page.goto("/no-such-page")
    found["404 signed in"] = problems(page)

    assert {screen: issues for screen, issues in found.items() if issues} == {}
