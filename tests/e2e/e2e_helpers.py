"""Step 7.4: steps the browser tests share."""

from playwright.sync_api import Page, expect


def sign_in(page: Page, email: str, password: str) -> None:
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Sign in").click()


def load_sample(page: Page, name: str) -> None:
    page.get_by_label("Sample applicant", exact=True).select_option(name)
    page.get_by_role("button", name="Load sample").click()
    expect(page.locator("#assess-status")).to_contain_text("Loaded")
