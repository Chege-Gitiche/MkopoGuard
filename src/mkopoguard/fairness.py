"""Step 4.4: the fairness check. Does the model treat groups of applicants differently?

Gender, age group and country are NEVER model inputs (age is, as a number, but the groups
are only for this audit). They are kept beside the data only to check the decisions.

For each group, using the cost-based decisions (approve / decline):

  approval_rate  share of applicants approved
  recall         share of applicants who went on to default that the model declined
  fpr            share of applicants who repaid that the model declined: good borrowers turned
                 away, the error that harms applicants (false positive rate)

Each group is compared with the reference group, the one approved most often:

  approval_ratio  approval rate / the reference's. Below 0.80 is the "four-fifths rule"
                  sign of adverse impact (US EEOC, 29 CFR 1607.4(D)): a common benchmark,
                  not Kenyan law, and only a sign, not a verdict
  fpr_gap         fpr - the reference's: how many more good borrowers (per 100) are declined

The data is synthetic, so we also know every applicant's TRUE probability of default. The
"oracle" decisions apply the same thresholds to the true probability: what a model with
perfect information would do. Groups really differ in risk (through income, work, savings and
statement access, never through gender itself), so some gap is expected even from the
oracle. The EXCESS gap (model gap - oracle gap) is the part the model itself adds.

Flags (the rules this audit applies; evaluation plan, step 4.4):
  adverse_impact  approval_ratio < 0.80
  fpr_flag        fpr_gap > 5 points, and its interval is entirely above zero
  model_adds_gap  the excess approval or fpr gap has an interval entirely above zero
Intervals are bootstrap intervals, widened for the number of groups compared (Bonferroni),
as in step 3.9.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

FOUR_FIFTHS = 0.80
MAX_FPR_GAP = 0.05
AGE_BINS = (17, 24, 34, 49, 200)
AGE_LABELS = ("18-24", "25-34", "35-49", "50+")
ATTRIBUTES = ("gender", "age_group", "country")
NEVER_INPUTS = ("is_female", "country_code")  # used by this audit only


def audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The groups each applicant belongs to, as readable labels."""
    female = frame["is_female"].astype("Int64")
    if female.isna().any():
        raise ValueError("is_female has missing values; every applicant needs a gender to audit")
    return pd.DataFrame(
        {
            "gender": np.where(female == 1, "female", "male"),
            "age_group": pd.cut(frame["age"], AGE_BINS, labels=AGE_LABELS),
            "country": frame["country_code"].astype(str),
        },
        index=frame.index,
    )


# --- rates per group -------------------------------------------------------------------------


def _rates(codes: np.ndarray, k: int, y: np.ndarray, decline: np.ndarray) -> dict:
    """Approval rate, recall and fpr for each of k groups (codes 0..k-1), in one pass."""
    n = np.bincount(codes, minlength=k).astype(float)
    bad = np.bincount(codes, weights=y, minlength=k)
    declined = np.bincount(codes, weights=decline, minlength=k)
    caught = np.bincount(codes, weights=y * decline, minlength=k)  # defaulters declined
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "n": n,
            "default_rate": bad / n,
            "approval_rate": 1 - declined / n,
            "recall": caught / bad,
            "fpr": (declined - caught) / (n - bad),
        }


def _gaps(r: dict, ref: int, oracle: dict | None = None) -> dict:
    """Each group against the reference group (and, if given, the same for the oracle)."""
    ref_approval = r["approval_rate"][ref]
    out = {
        # a resample where the reference approved nobody has no ratio (NaN, skipped)
        "approval_ratio": (
            r["approval_rate"] / ref_approval
            if ref_approval > 0
            else np.full_like(r["approval_rate"], np.nan)
        ),
        "approval_gap": ref_approval - r["approval_rate"],  # + = approved less
        "fpr_gap": r["fpr"] - r["fpr"][ref],  # + = more good borrowers declined
        "recall_gap": r["recall"] - r["recall"][ref],
    }
    if oracle is not None:
        oracle_approval_gap = oracle["approval_rate"][ref] - oracle["approval_rate"]
        oracle_fpr_gap = oracle["fpr"] - oracle["fpr"][ref]
        out["excess_approval_gap"] = out["approval_gap"] - oracle_approval_gap
        out["excess_fpr_gap"] = out["fpr_gap"] - oracle_fpr_gap
    return out


def bonferroni_level(n_groups: int, alpha: float = 0.05) -> float:
    """Interval level when n_groups - 1 groups are each compared with the reference."""
    return 1 - alpha / max(n_groups - 1, 1)


def audit(
    groups,
    y,
    decline,
    oracle_decline=None,
    reference: str | None = None,
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per group: rates, gaps to the reference, intervals and flags.

    groups          the group label of each applicant
    y               1 = defaulted
    decline         True = the model's decision was decline
    oracle_decline  the same rule applied to the true probability (optional)
    reference       the group to compare with (default: the one approved most often)
    """
    labels = pd.Categorical(groups)
    codes, names, k = labels.codes, list(labels.categories), len(labels.categories)
    if (codes < 0).any():
        raise ValueError("Every applicant needs a group (found missing labels)")
    y = np.asarray(y, dtype=float)
    decline = np.asarray(decline, dtype=float)
    oracle_decline = None if oracle_decline is None else np.asarray(oracle_decline, dtype=float)

    r = _rates(codes, k, y, decline)
    oracle = None if oracle_decline is None else _rates(codes, k, y, oracle_decline)
    ref = int(np.argmax(r["approval_rate"])) if reference is None else names.index(reference)
    table = pd.DataFrame(r | _gaps(r, ref, oracle), index=pd.Index(names, name="group"))
    if oracle is not None:
        table["oracle_approval_rate"] = oracle["approval_rate"]
        table["oracle_approval_ratio"] = oracle["approval_rate"] / oracle["approval_rate"][ref]
        table["oracle_fpr"] = oracle["fpr"]

    # bootstrap the gaps (the same resampled applicants for model and oracle)
    rng = np.random.default_rng(seed)
    keys = ["approval_ratio", "fpr_gap"] + (
        ["excess_approval_gap", "excess_fpr_gap"] if oracle is not None else []
    )
    draws = {key: [] for key in keys}
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        rb = _rates(codes[i], k, y[i], decline[i])
        ob = None if oracle is None else _rates(codes[i], k, y[i], oracle_decline[i])
        gb = _gaps(rb, ref, ob)
        for key in keys:
            draws[key].append(gb[key])
    tail = 100 * (1 - bonferroni_level(k)) / 2
    for key in keys:
        low, high = np.nanpercentile(np.array(draws[key]), [tail, 100 - tail], axis=0)
        table[f"{key}_low"], table[f"{key}_high"] = low, high

    table["reference"] = np.arange(k) == ref
    table["adverse_impact"] = table["approval_ratio"] < FOUR_FIFTHS
    table["fpr_flag"] = (table["fpr_gap"] > MAX_FPR_GAP) & (table["fpr_gap_low"] > 0)
    if oracle is not None:
        table["model_adds_gap"] = (table["excess_approval_gap_low"] > 0) | (
            table["excess_fpr_gap_low"] > 0
        )
    flag_cols = [c for c in ("adverse_impact", "fpr_flag", "model_adds_gap") if c in table]
    table.loc[table["reference"], flag_cols] = False
    return table


def same_risk_gap(groups, decline, true_p, reference: str, bins: int = 10) -> pd.DataFrame:
    """Decline rates by group among applicants with the SAME true risk (deciles of true risk).

    If the model treated equal-risk applicants equally, every gap would be about zero.
    The last row ("average") weights each decile by the group's applicants in it.
    """
    data = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "decline": np.asarray(decline, dtype=float),
            "risk_band": pd.qcut(np.asarray(true_p), bins, labels=False, duplicates="drop") + 1,
        }
    )
    rates = data.pivot_table(index="risk_band", columns="group", values="decline", aggfunc="mean")
    counts = data.pivot_table(index="risk_band", columns="group", values="decline", aggfunc="size")
    gaps = rates.sub(rates[reference], axis=0)
    average = (gaps * counts).sum() / counts.sum()
    out = gaps.add_suffix("_gap")
    out.loc["average"] = average.add_suffix("_gap")
    return out.drop(columns=f"{reference}_gap")
