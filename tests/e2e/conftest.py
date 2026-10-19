"""Step 7.4: browser tests, in a real Chromium against a throwaway copy of the app.

    python -m pytest -m e2e              run them (they are left out of a plain pytest run)
    python -m pytest -m e2e --headed     watch them in a visible browser window

The app runs on a free port with its own temporary database and officer account
(mkopoguard/local_server.py); nothing touches your own mkopoguard.db.
"""

import re

import pytest
from e2e_helpers import sign_in
from playwright.sync_api import Page, expect

from mkopoguard.local_server import local_server


@pytest.fixture(scope="session")
def server():
    with local_server() as running:
        yield running


@pytest.fixture(scope="session")
def base_url(server):
    """pytest-playwright opens every page relative to this, so page.goto("/assess.html") works."""
    return server.url


@pytest.fixture
def officer(page: Page, server) -> Page:
    """A page signed in as the officer, on New assessment."""
    page.goto("/")
    sign_in(page, server.email, server.password)
    expect(page).to_have_url(re.compile(r"/assess\.html$"))
    return page
