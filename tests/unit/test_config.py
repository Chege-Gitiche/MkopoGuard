from mkopoguard import config


def test_pool_has_20_unique_countries():
    assert len(config.POOL_COUNTRIES) == 20
    assert len(set(config.POOL_COUNTRIES)) == 20


def test_target_country_is_in_pool():
    assert config.TARGET_COUNTRY in config.POOL_COUNTRIES


def test_country_codes_are_iso3():
    assert all(len(c) == 3 and c.isupper() for c in config.POOL_COUNTRIES)
