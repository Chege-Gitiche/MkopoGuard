"""Step 5.5: scoring one application, independent of the web layer (so it is easy to test).

    profile + statement -> check -> build features -> predict -> explain -> result

The model and the explainer are loaded once at startup (main.py) and passed in. Nothing here
touches the database; the route decides whether to save (the demo account never does).
"""

import math

from mkopoguard.api.schemas import ApplicantProfile, band_for
from mkopoguard.api.statement import Statement, build_inputs
from mkopoguard.bundle import Bundle, check_input
from mkopoguard.explain.reasons import Reasoner
from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES

REASON_FIELDS = ("code", "text", "direction", "impact_points", "strength")


class ScoringError(RuntimeError):
    """The model and the explanation disagree: a bug, never the applicant's fault."""


def _number(value):
    """JSON-safe: NaN becomes null (Postgres refuses NaN inside JSON)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def score(
    bundle: Bundle,
    reasoner: Reasoner,
    profile: ApplicantProfile,
    statement: Statement | None,
) -> dict:
    """Everything the API returns and saves about one application (except id and time)."""
    inputs = build_inputs(profile, statement)
    problems = check_input(inputs, bundle.schema)
    if problems["errors"]:  # the schemas should make this impossible; refuse rather than guess
        raise ScoringError(f"Inputs the model can't score: {problems['errors']}")
    result = bundle.decide(inputs).iloc[0]
    probability = float(result["probability_of_default"])
    explained = reasoner.explain_rows(inputs)[0]
    if abs(explained["probability_of_default"] - probability) > 1e-9:
        raise ScoringError("The explanation doesn't match the model's probability")

    def reasons_of(items):
        return [{k: item[k] for k in REASON_FIELDS} for item in items]

    other = explained["other_factors"]
    features = (
        {name: _number(inputs.loc[0, name]) for name in STATEMENT_FEATURES}
        if statement is not None
        else None
    )
    return {
        "applicant_ref": profile.applicant_ref,
        "probability_of_default": probability,
        "band": band_for(probability),
        "decision": str(result["decision"]),
        "threshold": float(result["threshold"]),
        "term_months": profile.term_months,
        "has_statement": statement is not None,
        "reasons": {
            "raising": reasons_of(explained["raising"]),
            "lowering": reasons_of(explained["lowering"]),
            "other_factors": {
                "impact_points": float(other["impact_points"]),
                "material": bool(other["material"]),
            },
        },
        "warnings": (statement.warnings if statement else []) + problems["warnings"],
        "model_version": bundle.manifest["version"],
        # What is saved as the prediction's inputs: the answers and the statement's FEATURES,
        # never the raw transactions (data minimisation: they are personal financial records)
        "inputs": {
            "profile": profile.model_dump(mode="json"),
            "statement_features": features,
            "statement_transactions": len(statement.transactions) if statement else 0,
        },
    }
