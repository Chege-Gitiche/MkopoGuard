"""Step 5.4: sample applications for the demo: profiles + M-Pesa statements in the API format.

Run from the project root:  python scripts/make_samples.py

Picks 10 VALIDATION applicants (the test set stays closed): 8 with a statement, two from each
risk band, and 2 thin-file applicants (no statement). For each it writes
  samples/profiles/<name>.json     the profile, exactly as POST /predict accepts it
  samples/statements/<name>.csv    the statement in the fixed upload format (holders only)
and samples/manifest.json with the probability and decision the model must give for each.

Before writing, every sample goes through the API's own path (validate the profile, check the
statement, build the features, score) and must match the training data's features exactly, so
the samples double as an end-to-end test (tests/model/test_samples.py).
"""

import json
import shutil

import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.api.schemas import ApplicantProfile, band_for
from mkopoguard.api.statement import COLUMNS, build_inputs, parse_statement
from mkopoguard.bundle import load_bundle
from mkopoguard.models.experiment import load_training_data

SAMPLES = config.PROJECT_ROOT / "samples"
SEED = 2026
PER_BAND = 2
THIN_FILE = 2


def profile_of(row: pd.Series, application_date, ref: str) -> dict:
    def yes_no(value):
        return None if pd.isna(value) else bool(value)

    def category(value):
        return None if pd.isna(value) else str(value)

    return {
        "applicant_ref": ref,
        "application_date": application_date.date().isoformat(),
        "age": int(row["age"]),
        "income_quintile": int(row["income_quintile"]),
        "in_workforce": bool(row["in_workforce"]),
        "education_level": category(row["education_level"]),
        "loan_amount_kes": int(row["loan_amount_kes"]),
        "term_months": int(row["term_months"]),
        "saved_past_year": bool(row["saved_past_year"]),
        "saved_in_savings_club": yes_no(row["saved_in_savings_club"]),
        "borrowed_past_year": bool(row["borrowed_past_year"]),
        "borrowed_from_fin_institution": yes_no(row["borrowed_from_fin_institution"]),
        "borrowed_from_family_friends": yes_no(row["borrowed_from_family_friends"]),
        "borrowed_for_medical": yes_no(row["borrowed_for_medical"]),
        "emergency_fund_source": category(row["emergency_fund_source"]),
        "worried_about_bills": category(row["worried_about_bills"]),
    }


def pick(validation: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    frame = validation.assign(p=p, band=[band_for(x) for x in p])
    rng = np.random.default_rng(SEED)
    picks = []
    holders = frame[frame["has_statement"] == 1]
    for band in ("low", "medium", "high", "very_high"):
        options = holders[holders["band"] == band]
        picks.append(options.iloc[rng.choice(len(options), PER_BAND, replace=False)])
    thin = frame[frame["has_statement"] == 0]
    picks.append(thin.iloc[rng.choice(len(thin), THIN_FILE, replace=False)])
    return pd.concat(picks)


def main() -> None:
    champion = load_bundle()
    _, validation = load_training_data()
    chosen = pick(validation, champion.predict_proba(validation))
    dates = pd.read_parquet(
        config.DATA_PROCESSED / "applicants.parquet", columns=["applicant_id", "application_date"]
    ).set_index("applicant_id")["application_date"]
    transactions = pd.read_parquet(config.DATA_PROCESSED / "transactions.parquet")
    transactions = transactions[transactions["applicant_id"].isin(chosen["applicant_id"])]

    shutil.rmtree(SAMPLES, ignore_errors=True)
    (SAMPLES / "profiles").mkdir(parents=True)
    (SAMPLES / "statements").mkdir()
    manifest = []
    for i, (_, row) in enumerate(chosen.iterrows(), start=1):
        has = row["has_statement"] == 1
        name = f"sample_{i:02d}_{row['band'] if has else 'thin_file'}"
        profile = profile_of(row, dates[row["applicant_id"]], f"SAMPLE-{i:02d}")
        validated = ApplicantProfile.model_validate_json(json.dumps(profile))
        statement = None
        if has:
            own = transactions[transactions["applicant_id"] == row["applicant_id"]]
            csv_text = own[list(COLUMNS)].to_csv(index=False, lineterminator="\n")
            (SAMPLES / "statements" / f"{name}.csv").write_text(csv_text, encoding="utf-8")
            statement = parse_statement(csv_text.encode("utf-8"), validated.application_date)
        inputs = build_inputs(validated, statement)
        p = float(champion.predict_proba(inputs)[0])
        if abs(p - row["p"]) > config.GOLDEN_TOLERANCE:  # the API path must match training
            raise SystemExit(f"{name}: API path gives {p}, training data gives {row['p']}")
        decision = champion.decide(inputs)["decision"].iloc[0]
        (SAMPLES / "profiles" / f"{name}.json").write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        manifest.append(
            {
                "name": name,
                "profile": f"profiles/{name}.json",
                "statement": f"statements/{name}.csv" if has else None,
                "probability_of_default": p,
                "band": band_for(p),
                "decision": decision,
                "model_version": config.MODEL_VERSION,
            }
        )
        rows = len(own) if has else 0
        print(f"{name:<28} {p:6.1%}  {decision:<8} {rows:>4} transactions")
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} samples to {SAMPLES.relative_to(config.PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
