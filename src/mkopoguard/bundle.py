"""Step 3.13: the model bundle, the one folder the API (phase 4) needs to score applicants.

models/champion/
  model.skops            the fitted pipeline (preprocessing + model), in skops' safe format
  input_schema.json      every input column: kind, allowed values or the range seen in training
  decision_policy.json   the cost-based thresholds from step 3.11
  golden_examples.json   applicants with their exact expected probability and decision
  model_card.md          what the model is for, how well it works, and its limits
  manifest.json          version, provenance, library versions and a SHA-256 of every file

load_bundle() refuses a bundle whose files don't match their checksums (corrupted or edited),
warns if the installed scikit-learn differs from the one that built it, and can replay the
golden examples to prove the model behaves exactly as it did when it was packaged.
"""

import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio

from mkopoguard import config
from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC, SKOPS_TRUSTED_TYPES
from mkopoguard.models import decision as dec

FILES = {
    "model": "model.skops",
    "schema": "input_schema.json",
    "policy": "decision_policy.json",
    "golden": "golden_examples.json",
    "card": "model_card.md",
}
MANIFEST = "manifest.json"
LIBRARIES = ("sklearn", "numpy", "pandas", "skops")


class BundleError(Exception):
    """The bundle is incomplete, corrupted, or doesn't reproduce its golden examples."""


# --- helpers ----------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn
    import skops

    modules = {"sklearn": sklearn, "numpy": numpy, "pandas": pandas, "skops": skops}
    return {"python": platform.python_version()} | {n: modules[n].__version__ for n in LIBRARIES}


def _write_text(path: Path, text: str) -> None:
    """Always "\n" line endings, even on Windows, so checksums match on every machine."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, data) -> None:
    _write_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def _clean(value):
    """A JSON-safe version of one input value (NaN / pd.NA become null)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


# --- input schema -------------------------------------------------------------------------


def input_schema(pipeline, development: pd.DataFrame) -> dict:
    """What every input column may contain, learned from the fitted pipeline and its data."""
    prep = pipeline.named_steps["prep"]
    columns = list(prep.feature_names_in_)
    categories = {}
    for name, transformer, cols in prep.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["one_hot"]
            categories = {
                c: [str(v) for v in cats] for c, cats in zip(cols, encoder.categories_, strict=True)
            }
    schema = {}
    for col in columns:
        values = development[col]
        entry = {"nullable": bool(values.isna().any())}
        if col in CATEGORICAL:
            entry |= {"kind": "categorical", "allowed": categories.get(col, [])}
        elif col in BINARY:
            entry |= {"kind": "binary", "allowed": [0, 1]}
        elif col in NUMERIC:
            numbers = pd.to_numeric(values, errors="coerce").dropna()
            entry |= {"kind": "numeric", "min": float(numbers.min()), "max": float(numbers.max())}
            if col == "term_months":  # only the terms the decision policy has thresholds for
                entry["allowed"] = [int(t) for t in config.TERM_OPTIONS]
        else:
            raise BundleError(f"Column {col!r} is not in NUMERIC, BINARY or CATEGORICAL")
        schema[col] = entry
    return {"columns": columns, "fields": schema}


def check_input(frame: pd.DataFrame, schema: dict) -> dict[str, list[str]]:
    """Problems with applicants before scoring them.

    errors   - the model can't score this: a missing column, a null where none is allowed,
               a value of the wrong kind
    warnings - the model can score it, but it is unusual: a category or a number outside what
               the model saw in training (it will be handled, but treat the score with care)
    """
    errors, warns = [], []
    missing = [c for c in schema["columns"] if c not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    for col in schema["columns"]:
        if col not in frame.columns:
            continue
        field, values = schema["fields"][col], frame[col]
        present = values.dropna()
        if not field["nullable"] and len(present) < len(values):
            errors.append(f"{col}: empty values are not allowed")
        if field["kind"] == "categorical":
            unknown = sorted(set(present.astype(str)) - set(field["allowed"]))
            if unknown:
                warns.append(f"{col}: unseen categories {unknown}")
        else:
            numbers = pd.to_numeric(present, errors="coerce")
            if numbers.isna().any():
                errors.append(f"{col}: values must be numbers")
                continue
            if field["kind"] == "binary" and not numbers.isin([0, 1]).all():
                errors.append(f"{col}: values must be 0 or 1")
            if "allowed" in field and field["kind"] == "numeric":
                if not numbers.isin(field["allowed"]).all():
                    errors.append(f"{col}: values must be one of {field['allowed']}")
            if field["kind"] == "numeric":
                outside = (numbers < field["min"]) | (numbers > field["max"])
                if outside.any():
                    warns.append(
                        f"{col}: {int(outside.sum())} value(s) outside the training range "
                        f"{field['min']:g} to {field['max']:g}"
                    )
    return {"errors": errors, "warnings": warns}


# --- the bundle -------------------------------------------------------------------------------


@dataclass
class Bundle:
    model: object
    schema: dict
    policy: dict
    manifest: dict
    golden: list[dict]
    path: Path

    @property
    def columns(self) -> list[str]:
        return self.schema["columns"]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of default for each applicant. Refuses input with errors."""
        problems = check_input(frame, self.schema)
        if problems["errors"]:
            raise ValueError("Can't score these applicants: " + "; ".join(problems["errors"]))
        return self.model.predict_proba(frame[self.columns])[:, 1]

    def decide(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Probability, the threshold for the loan's term, and approve / decline."""
        p = self.predict_proba(frame)
        terms = frame["term_months"].astype(int).to_numpy()
        thresholds = self.policy["thresholds_by_term_months"]
        return pd.DataFrame(
            {
                "probability_of_default": p,
                "threshold": [thresholds[t] for t in terms],
                "decision": [
                    dec.decide(pi, t, self.policy) for pi, t in zip(p, terms, strict=True)
                ],
            },
            index=frame.index,
        )

    def verify_golden(self, tolerance: float = config.GOLDEN_TOLERANCE) -> float:
        """Re-score the golden examples; fail if any probability or decision has changed."""
        frame = golden_frame(self.golden, self.schema)
        result = self.decide(frame)
        expected = np.array([g["probability_of_default"] for g in self.golden])
        gap = float(np.max(np.abs(result["probability_of_default"].to_numpy() - expected)))
        if gap > tolerance:
            raise BundleError(f"Golden examples changed by up to {gap:.3g} (allowed {tolerance})")
        changed = [
            g["applicant_id"]
            for g, d in zip(self.golden, result["decision"], strict=True)
            if g["decision"] != d
        ]
        if changed:
            raise BundleError(f"Golden decisions changed for {changed}")
        return gap


def golden_frame(golden: list[dict], schema: dict) -> pd.DataFrame:
    """The golden examples as a table the model can score (nulls become NaN)."""
    rows = [g["inputs"] for g in golden]
    frame = pd.DataFrame(rows, columns=schema["columns"])
    for col, field in schema["fields"].items():
        if field["kind"] != "categorical":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def pick_golden(frame: pd.DataFrame, per_group: int = 4, seed: int = 0) -> pd.DataFrame:
    """A few applicants from every loan term x statement / thin-file combination."""
    picks = [
        group.sample(min(per_group, len(group)), random_state=seed)
        for _, group in frame.groupby(["term_months", "has_statement"])
    ]
    return pd.concat(picks)


def build_bundle(
    pipeline,
    development: pd.DataFrame,
    golden_source: pd.DataFrame,
    out_dir: Path,
    policy: dict,
    model_card: str,
    provenance: dict,
) -> Path:
    """Write every bundle file, then the manifest with their checksums."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = input_schema(pipeline, development)

    sio.dump(pipeline, out_dir / FILES["model"])
    _write_json(out_dir / FILES["schema"], schema)
    _write_json(out_dir / FILES["policy"], policy)

    examples = pick_golden(golden_source)
    p = pipeline.predict_proba(examples[schema["columns"]])[:, 1]
    golden = [
        {
            "applicant_id": str(row["applicant_id"]),
            "inputs": {c: _clean(row[c]) for c in schema["columns"]},
            "probability_of_default": float(pi),
            "decision": dec.decide(float(pi), int(row["term_months"]), policy),
        }
        for (_, row), pi in zip(examples.iterrows(), p, strict=True)
    ]
    _write_json(out_dir / FILES["golden"], golden)
    _write_text(out_dir / FILES["card"], model_card)

    manifest = {
        "name": config.MODEL_NAME,
        "version": config.MODEL_VERSION,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_input_columns": len(schema["columns"]),
        "n_golden_examples": len(golden),
        "environment": library_versions(),
        **provenance,
        "files": {name: sha256(out_dir / name) for name in FILES.values()},
    }
    _write_json(out_dir / MANIFEST, manifest)
    return out_dir


def load_bundle(path: Path = config.BUNDLE_DIR, verify_golden: bool = True) -> Bundle:
    """Load a bundle after checking every file against its checksum."""
    path = Path(path)
    if not (path / MANIFEST).exists():
        raise BundleError(f"No {MANIFEST} in {path}; run scripts/package_model.py first")
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        file = path / name
        if not file.exists():
            raise BundleError(f"{name} is missing from the bundle")
        if sha256(file) != expected:
            raise BundleError(f"{name} doesn't match its checksum: corrupted or edited")

    built, installed = manifest["environment"]["sklearn"], library_versions()["sklearn"]
    if built.split(".")[:2] != installed.split(".")[:2]:
        warnings.warn(
            f"Bundle built with scikit-learn {built}, but {installed} is installed; "
            "check the golden examples still pass",
            stacklevel=2,
        )

    model = sio.load(path / FILES["model"], trusted=SKOPS_TRUSTED_TYPES)
    schema = json.loads((path / FILES["schema"]).read_text(encoding="utf-8"))
    policy = dec.load_policy(path / FILES["policy"])
    golden = json.loads((path / FILES["golden"]).read_text(encoding="utf-8"))
    bundle = Bundle(model, schema, policy, manifest, golden, path)
    if verify_golden:
        bundle.verify_golden()
    return bundle
