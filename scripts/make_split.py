"""Step 2.6: create the one-and-only train/validation/test split.

Run from the project root:  python scripts/make_split.py

Splitting happens ONCE. If a split file already exists and a fresh split would differ,
the script stops rather than overwrite it. Use --force only if the applicants themselves
changed (e.g. the simulator was rerun with new rules), and say so in the devlog.
"""

import argparse

import pandas as pd

from mkopoguard import config
from mkopoguard.data.split import make_splits, split_summary, write_once

FEATURES_PATH = config.DATA_PROCESSED / "features.parquet"
MESSAGES = {
    "created": f"Saved the split to {config.SPLIT_FILE}",
    "unchanged": "Split file already exists and matches. Nothing to do.",
    "overwritten": "Split file overwritten (--force). Record why in the devlog.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the train/validation/test split.")
    parser.add_argument("--force", action="store_true", help="overwrite a different split")
    args = parser.parse_args()

    table = pd.read_parquet(FEATURES_PATH)
    splits = make_splits(table)
    try:
        outcome = write_once(splits, force=args.force)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error

    print(MESSAGES[outcome])
    print(split_summary(table, splits).round(3).to_string())


if __name__ == "__main__":
    main()
