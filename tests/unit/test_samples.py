"""Step 5.4: the demo samples (committed in samples/) go through the API's whole input path
and score exactly what the model gave when they were made. Runs in CI."""

import json

import pytest

from mkopoguard import config
from mkopoguard.api.schemas import ApplicantProfile, band_for
from mkopoguard.api.statement import MAX_BYTES, build_inputs, parse_statement
from mkopoguard.bundle import load_bundle

SAMPLES = config.PROJECT_ROOT / "samples"
pytestmark = pytest.mark.skipif(
    not (SAMPLES / "manifest.json").exists(), reason="samples not made yet"
)
MANIFEST = (
    json.loads((SAMPLES / "manifest.json").read_text()) if (SAMPLES / "manifest.json").exists()
    else []
)  # fmt: skip


@pytest.fixture(scope="module")
def champion():
    return load_bundle()


def test_there_are_samples_for_every_band_and_thin_file():
    assert len(MANIFEST) >= 5
    assert {s["band"] for s in MANIFEST if s["statement"]} == {"low", "medium", "high", "very_high"}
    assert any(s["statement"] is None for s in MANIFEST)
    assert {s["decision"] for s in MANIFEST} == {"approve", "decline"}


@pytest.mark.parametrize("sample", MANIFEST, ids=[s["name"] for s in MANIFEST])
def test_sample_scores_as_recorded(champion, sample):
    profile = ApplicantProfile.model_validate_json((SAMPLES / sample["profile"]).read_text())
    statement = None
    if sample["statement"]:
        data = (SAMPLES / sample["statement"]).read_bytes()
        assert len(data) <= MAX_BYTES
        statement = parse_statement(data, profile.application_date)
    inputs = build_inputs(profile, statement)
    result = champion.decide(inputs).iloc[0]
    assert result["probability_of_default"] == pytest.approx(
        sample["probability_of_default"], abs=config.GOLDEN_TOLERANCE
    )
    assert result["decision"] == sample["decision"]
    assert band_for(result["probability_of_default"]) == sample["band"]
