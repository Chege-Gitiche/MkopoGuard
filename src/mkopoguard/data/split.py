"""Step 2.6: split applicants once into train (70%), validation (15%) and test (15%).

The split is stratified by country x defaulted x has_statement (80 groups), so every part
has the same mix of countries, default rate and thin-file share as the whole pool.

Only applicant IDs are saved (data/splits/applicant_split.csv, committed to Git), so the
split stays fixed even if features are rebuilt. The test set is locked: load_split()
refuses to return it unless you pass allow_test=True, which should happen exactly once,
at the very end of Phase 3 (step 3.12).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from mkopoguard import config

SPLITS = ("train", "validation", "test")
SIZES = {"train": 0.70, "validation": 0.15, "test": 0.15}
STRATIFY = ["country_code", "defaulted", "has_statement"]


def make_splits(table: pd.DataFrame, seed: int = config.SPLIT_SEED) -> pd.DataFrame:
    """Return a table of applicant_id and split, sorted by applicant_id.

    The result depends only on the applicants and the seed, never on row order.
    """
    t = table[["applicant_id", *STRATIFY]].sort_values("applicant_id").reset_index(drop=True)
    groups = t[STRATIFY].astype(str).agg("|".join, axis=1)

    held_out = SIZES["validation"] + SIZES["test"]
    train, rest = train_test_split(t.index, test_size=held_out, stratify=groups, random_state=seed)
    test_share_of_rest = SIZES["test"] / held_out
    validation, test = train_test_split(
        rest, test_size=test_share_of_rest, stratify=groups.loc[rest], random_state=seed
    )

    split = pd.Series("train", index=t.index)
    split.loc[validation] = "validation"
    split.loc[test] = "test"
    return pd.DataFrame({"applicant_id": t["applicant_id"], "split": split})


def save_splits(splits: pd.DataFrame, path=config.SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False, lineterminator="\n")


def write_once(splits: pd.DataFrame, path=config.SPLIT_FILE, force: bool = False) -> str:
    """Save the split unless a DIFFERENT one already exists. Returns what happened.

    Re-splitting after seeing results quietly leaks the test set into your decisions,
    so an existing split is only replaced with force=True.
    """
    if not path.exists():
        save_splits(splits, path)
        return "created"
    if read_splits(path).astype(str).equals(splits.astype(str).reset_index(drop=True)):
        return "unchanged"
    if not force:
        raise FileExistsError(
            f"{path} exists and differs from a fresh split. Not overwriting."
            " Use --force only if the applicants themselves changed."
        )
    save_splits(splits, path)
    return "overwritten"


def read_splits(path=config.SPLIT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No split file at {path}. Run: python scripts/make_split.py")
    return pd.read_csv(path, dtype={"applicant_id": str, "split": str})


def load_split(table: pd.DataFrame, name: str, allow_test: bool = False, path=config.SPLIT_FILE):
    """Rows of `table` in one split. The test set needs allow_test=True (step 3.12 only)."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; choose from {SPLITS}")
    if name == "test" and not allow_test:
        raise PermissionError(
            "The test set is locked until step 3.12. Tune on 'train' and 'validation' only."
        )
    ids = read_splits(path)
    wanted = set(ids.loc[ids["split"] == name, "applicant_id"])
    return table[table["applicant_id"].isin(wanted)].reset_index(drop=True)


def split_summary(table: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Size and mix of each split, for a quick visual check."""
    t = table.merge(splits, on="applicant_id")
    summary = t.groupby("split").agg(
        applicants=("applicant_id", "size"),
        default_rate=("defaulted", "mean"),
        with_statement=("has_statement", "mean"),
        kenyans=("country_code", lambda s: int((s == config.TARGET_COUNTRY).sum())),
        countries=("country_code", "nunique"),
    )
    summary.insert(1, "share", summary["applicants"] / len(t))
    return summary.reindex(list(SPLITS))
