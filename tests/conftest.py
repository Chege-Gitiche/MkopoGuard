"""Shared test setup.

API tests (step 5.2+) never touch the developer's own mkopoguard.db: for the whole test run,
MKOPOGUARD_DATABASE_URL points at a fresh SQLite file in a temporary folder, migrated to the
newest schema.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def test_database_url(tmp_path_factory):
    from mkopoguard.api.db import upgrade

    url = f"sqlite:///{(tmp_path_factory.mktemp('db') / 'test.db').as_posix()}"
    upgrade(url)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MKOPOGUARD_DATABASE_URL", url)
        yield url
