"""Step 5.4 on real (synthetic) statements: uploading a statement as a CSV gives exactly the
features, and the score, the model was trained on."""

import json

import numpy as np
import pandas as pd
import pytest

from mkopoguard import bundle, config
from mkopoguard.api.schemas import ApplicantProfile
from mkopoguard.api.statement import COLUMNS, build_inputs, parse_statement

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "transactions.parquet").exists(), reason="data not built"
)


def test_csv_upload_reproduces_the_training_features_and_score():
    import importlib.util

    from mkopoguard.models.experiment import load_training_data

    path = config.PROJECT_ROOT / "scripts" / "make_samples.py"
    spec = importlib.util.spec_from_file_location("make_samples", path)
    samples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(samples)

    champion = bundle.load_bundle()
    _, validation = load_training_data()
    holders = validation[validation["has_statement"] == 1].sample(200, random_state=0)
    dates = pd.read_parquet(
        config.DATA_PROCESSED / "applicants.parquet", columns=["applicant_id", "application_date"]
    ).set_index("applicant_id")["application_date"]
    tx = pd.read_parquet(config.DATA_PROCESSED / "transactions.parquet")
    tx = tx[tx["applicant_id"].isin(holders["applicant_id"])]
    rows = []
    for _, row in holders.iterrows():
        answers = samples.profile_of(row, dates[row["applicant_id"]], "REF-1")
        profile = ApplicantProfile.model_validate_json(json.dumps(answers))
        own = tx[tx["applicant_id"] == row["applicant_id"]][list(COLUMNS)]
        statement = parse_statement(own.to_csv(index=False).encode(), profile.application_date)
        rows.append(build_inputs(profile, statement))
    uploaded = pd.concat(rows, ignore_index=True)
    stmt = [c for c in champion.columns if c.startswith("stmt_")]
    expected = holders.reset_index(drop=True)
    np.testing.assert_allclose(
        uploaded[stmt].astype(float), expected[stmt].astype(float), atol=1e-9
    )
    gap = np.abs(champion.predict_proba(uploaded) - champion.predict_proba(expected)).max()
    assert gap <= config.GOLDEN_TOLERANCE
