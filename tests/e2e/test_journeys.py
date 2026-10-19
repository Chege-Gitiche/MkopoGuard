"""Step 7.4: the officer's journeys, end to end in a real browser (FR-23).

Test cases TC-07, TC-42, TC-43 and TC-51 in docs/test_plan.md. Each test starts in a fresh
browser with no cookies; the app and its database live for the whole run.
"""

import re

import pytest
from e2e_helpers import load_sample, sign_in
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_assess_a_sample_find_it_in_history_and_log_out(officer: Page):
    """TC-51: login -> load sample 07 -> assess -> result -> history -> saved record -> log out."""
    page = officer
    load_sample(page, "sample_07_very_high")
    expect(page.get_by_label("Applicant reference")).to_have_value("SAMPLE-07")
    expect(page.locator("#assess-status")).to_contain_text("with its statement")

    page.get_by_role("button", name="Assess risk").click()
    expect(page).to_have_url(re.compile(r"/result\.html\?id=\d+$"))
    expect(page.locator("#probability")).to_have_text("49.1%")
    expect(page.locator("#band")).to_have_text("Very high")
    expect(page.locator("#decision")).to_have_text("Decline")
    expect(page.get_by_role("heading", name="What raised the risk")).to_be_visible()

    page.get_by_role("link", name="History").click()
    row = page.get_by_role("row").filter(has_text="SAMPLE-07").first
    expect(row).to_contain_text("49.1%")
    row.get_by_role("link", name="View SAMPLE-07").click()
    expect(page).to_have_url(re.compile(r"/detail\.html\?id=\d+$"))
    expect(page.locator("#probability")).to_have_text("49.1%")
    expect(page.locator("#answers")).to_contain_text("22")  # the applicant's age, among the answers

    page.get_by_role("button", name="Log out").click()
    expect(page).to_have_url(re.compile(r"/\?reason=signed-out$"))
    expect(page.locator("#login-status")).to_have_text("You have logged out.")
    page.goto("/history.html")  # the session really is gone
    expect(page).to_have_url(re.compile(r"/\?reason=expired&next="))


def test_a_signed_out_officer_comes_back_to_the_page_they_asked_for(page: Page, server):
    """TC-07: History page 2 without a session -> login -> back on History page 2."""
    page.goto("/history.html?page=2")
    expect(page).to_have_url(re.compile(r"/\?reason=expired&next=%2Fhistory\.html%3Fpage%3D2$"))
    expect(page.locator("#login-status")).to_contain_text("Sign in to continue")
    with page.expect_navigation(url=re.compile(r"/history\.html\?page=2$")):
        sign_in(page, server.email, server.password)
    # (with fewer than 21 records, History then moves on to its last page, as designed)
    expect(page.get_by_role("heading", name="History", exact=True)).to_be_visible()


def test_a_wrong_password_keeps_the_officer_on_the_login_page(page: Page, server):
    """TC-03 in the browser: the error is shown and the password box is emptied."""
    page.goto("/")
    sign_in(page, server.email, "not the right password at all")
    expect(page.locator("#login-error")).to_contain_text("Incorrect email or password")
    expect(page.get_by_label("Password")).to_have_value("")
    expect(page).to_have_url(re.compile(r"/$"))


def test_the_browser_lists_every_problem_and_sends_nothing(officer: Page):
    """TC-42: an empty form is checked in the browser; nothing reaches /predict."""
    page = officer
    sent = []
    page.on("request", lambda request: sent.append(request.url))
    page.get_by_role("button", name="Assess risk").click()
    summary = page.locator("#form-errors")
    expect(summary).to_be_visible()
    expect(summary).to_be_focused()
    expect(summary.get_by_role("listitem")).to_have_count(9)  # 9 required answers
    expect(page.locator("#age-error")).to_have_text("Enter an age from 18 to 100.")
    expect(page.get_by_label("Age", exact=True)).to_have_attribute("aria-invalid", "true")
    assert not [url for url in sent if "/predict" in url]


def test_a_statement_the_server_refuses_is_explained_next_to_its_field(officer: Page):
    """TC-43: a one-row statement passes the browser checks; the server's reason is shown."""
    page = officer
    load_sample(page, "sample_07_very_high")
    one_row = (
        b"timestamp,type,direction,amount_kes,balance_after\n"
        b"2025-04-01 10:00:00,airtime,out,50,950\n"
    )
    page.locator("#statement").set_input_files(
        {"name": "one_row.csv", "mimeType": "text/csv", "buffer": one_row}
    )
    page.get_by_role("button", name="Assess risk").click()
    expect(page.locator("#statement-error")).to_contain_text("at least 10")
    expect(page.locator("#form-errors")).to_contain_text("Only 1 transactions")
    expect(page).to_have_url(re.compile(r"/assess\.html$"))
    expect(page.get_by_role("button", name="Assess risk")).to_be_enabled()
