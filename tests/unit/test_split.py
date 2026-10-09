import numpy as np
import pandas as pd
import pytest

from mkopoguard.data.split import (
    SPLITS,
    load_split,
    make_splits,
    read_splits,
    save_splits,
    write_once,
)

COUNTRIES = ["KEN", "UGA", "GHA", "TZA"]


def make_table(n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "applicant_id": [f"A-{i:05d}" for i in range(1, n + 1)],
            "country_code": rng.choice(COUNTRIES, n),
            "defaulted": (rng.random(n) < 0.2).astype("int8"),
            "has_statement": (rng.random(n) < 0.5).astype("int8"),
        }
    )


@pytest.fixture(scope="module")
def table():
    return make_table()


@pytest.fixture(scope="module")
def splits(table):
    return make_splits(table)


# --- sizes and coverage ---------------------------------------------------------


def test_every_applicant_is_in_exactly_one_split(table, splits):
    assert splits["applicant_id"].is_unique
    assert set(splits["applicant_id"]) == set(table["applicant_id"])
    assert set(splits["split"]) == set(SPLITS)


@pytest.mark.parametrize("name, share", [("train", 0.70), ("validation", 0.15), ("test", 0.15)])
def test_split_sizes(splits, name, share):
    assert (splits["split"] == name).mean() == pytest.approx(share, abs=0.005)


# --- stratification -------------------------------------------------------------


@pytest.mark.parametrize("column", ["defaulted", "has_statement"])
def test_each_split_has_the_same_mix_as_the_whole(table, splits, column):
    t = table.merge(splits, on="applicant_id")
    overall = t[column].mean()
    for name in SPLITS:
        assert t.loc[t["split"] == name, column].mean() == pytest.approx(overall, abs=0.02)


def test_every_country_appears_in_every_split(table, splits):
    t = table.merge(splits, on="applicant_id")
    for name in SPLITS:
        assert set(t.loc[t["split"] == name, "country_code"]) == set(COUNTRIES)


# --- reproducibility --------------------------------------------------------------


def test_same_seed_gives_the_same_split(table):
    pd.testing.assert_frame_equal(make_splits(table, seed=1), make_splits(table, seed=1))


def test_row_order_does_not_change_the_split(table):
    shuffled = table.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(make_splits(table), make_splits(shuffled))


def test_different_seed_gives_a_different_split(table):
    a = make_splits(table, seed=1)["split"]
    b = make_splits(table, seed=2)["split"]
    assert not a.equals(b)


# --- saving: split once, never silently overwrite ---------------------------------


def test_save_and_read_round_trip(splits, tmp_path):
    path = tmp_path / "split.csv"
    save_splits(splits, path)
    pd.testing.assert_frame_equal(read_splits(path).astype(str), splits.astype(str))


def test_write_once_creates_then_leaves_an_identical_split_alone(splits, tmp_path):
    path = tmp_path / "split.csv"
    assert write_once(splits, path) == "created"
    assert write_once(splits, path) == "unchanged"


def test_write_once_refuses_to_overwrite_a_different_split(table, tmp_path):
    path = tmp_path / "split.csv"
    write_once(make_splits(table, seed=1), path)
    with pytest.raises(FileExistsError, match="Not overwriting"):
        write_once(make_splits(table, seed=2), path)
    assert write_once(make_splits(table, seed=2), path, force=True) == "overwritten"


# --- the locked test set -----------------------------------------------------------


def test_test_set_is_locked_by_default(table, splits, tmp_path):
    path = tmp_path / "split.csv"
    save_splits(splits, path)
    with pytest.raises(PermissionError, match="locked"):
        load_split(table, "test", path=path)


def test_load_split_returns_exactly_that_split(table, splits, tmp_path):
    path = tmp_path / "split.csv"
    save_splits(splits, path)
    for name in SPLITS:
        rows = load_split(table, name, allow_test=True, path=path)
        expected = set(splits.loc[splits["split"] == name, "applicant_id"])
        assert set(rows["applicant_id"]) == expected


def test_unknown_split_name_fails_clearly(table, splits, tmp_path):
    path = tmp_path / "split.csv"
    save_splits(splits, path)
    with pytest.raises(ValueError, match="Unknown split"):
        load_split(table, "training", path=path)


def test_missing_split_file_explains_how_to_make_it(table, tmp_path):
    with pytest.raises(FileNotFoundError, match="make_split.py"):
        load_split(table, "train", path=tmp_path / "nope.csv")
