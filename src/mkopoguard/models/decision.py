"""Step 3.11: turn a probability of default into approve / decline, using the lender's costs.

For a loan of principal L and term T months, with a net margin m per month (fee minus the
lender's funding cost) and a loss given default LGD (share of principal lost):
- approve and the borrower repays:   the lender earns   m * T * L
- approve and the borrower defaults: the lender loses   LGD * L
- decline:                           nothing happens    0

Approving is worth it when the expected profit is positive:
    (1 - p) * m * T * L  >  p * LGD * L     <=>     p  <  m*T / (m*T + LGD)
So the threshold comes from the costs, not from the data, and L cancels out. Longer loans
earn more if repaid, so they can carry more risk. This only works if p is an honest
probability, which step 3.10 checked.

The rule is then checked against what actually happened: realised profit on out-of-fold CV
predictions (to compare policies) and once on the validation set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mkopoguard import config

NET_MARGIN = config.MONTHLY_FEE_RATE - config.MONTHLY_FUNDING_COST


def threshold(term_months, margin: float = NET_MARGIN, lgd: float = config.LOSS_GIVEN_DEFAULT):
    """Highest probability of default worth approving, for a loan of this term."""
    gain = margin * np.asarray(term_months, dtype=float)
    if margin <= 0:
        return np.zeros_like(gain)  # a loan that earns nothing is never worth the risk
    return gain / (gain + lgd)


def thresholds_by_term(margin: float = NET_MARGIN, lgd: float = config.LOSS_GIVEN_DEFAULT):
    return {int(t): float(threshold(t, margin, lgd)) for t in config.TERM_OPTIONS}


def cost_rule(p, term_months, margin: float = NET_MARGIN, lgd=config.LOSS_GIVEN_DEFAULT):
    """Approve (True) when the probability of default is below the loan's threshold."""
    return np.asarray(p) < threshold(term_months, margin, lgd)


def realised_profit(
    y, approve, amount, term_months, margin=NET_MARGIN, lgd=config.LOSS_GIVEN_DEFAULT
) -> np.ndarray:
    """Profit in KES for each applicant, given what actually happened (y = 1: defaulted)."""
    y, approve = np.asarray(y), np.asarray(approve, dtype=bool)
    amount, term = np.asarray(amount, dtype=float), np.asarray(term_months, dtype=float)
    outcome = np.where(y == 1, -lgd * amount, margin * term * amount)
    return np.where(approve, outcome, 0.0)


def expected_profit(
    p, approve, amount, term_months, margin=NET_MARGIN, lgd=config.LOSS_GIVEN_DEFAULT
) -> np.ndarray:
    """Profit in KES the model EXPECTS for each applicant (using p instead of the outcome)."""
    p, approve = np.asarray(p), np.asarray(approve, dtype=bool)
    amount, term = np.asarray(amount, dtype=float), np.asarray(term_months, dtype=float)
    outcome = (1 - p) * margin * term * amount - p * lgd * amount
    return np.where(approve, outcome, 0.0)


def policy_summary(name: str, y, approve, amount, term_months, **costs) -> dict:
    """Approval rate, defaults among approved, and profit for one way of deciding."""
    y, approve = np.asarray(y), np.asarray(approve, dtype=bool)
    profit = realised_profit(y, approve, amount, term_months, **costs)
    return {
        "policy": name,
        "approval_rate": float(approve.mean()),
        "default_rate_approved": float(y[approve].mean()) if approve.any() else float("nan"),
        "defaulters_caught": float(1 - approve[y == 1].mean()),
        "good_borrowers_declined": float(1 - approve[y == 0].mean()),
        "profit_kes": float(profit.sum()),
        "profit_per_1000_applicants_kes": float(profit.mean() * 1_000),
    }


def best_single_threshold(y, p, amount, term_months, grid=None, **costs) -> float:
    """The one threshold for all loans that would have earned the most (for comparison)."""
    grid = np.round(np.arange(0.01, 0.81, 0.01), 2) if grid is None else grid
    profits = [realised_profit(y, p < t, amount, term_months, **costs).sum() for t in grid]
    return float(grid[int(np.argmax(profits))])


def scale_check(y, p, amount, term_months, scales=None, **costs) -> pd.DataFrame:
    """Realised profit if every cost-based threshold were scaled up or down.

    If the probabilities are honest and the costs right, scale 1.0 should be (close to) best.
    """
    scales = np.round(np.arange(0.5, 1.51, 0.05), 2) if scales is None else scales
    base = threshold(term_months, **costs)
    rows = [
        {
            "scale": s,
            "profit_kes": realised_profit(
                y, np.asarray(p) < s * base, amount, term_months, **costs
            ).sum(),
        }
        for s in scales
    ]
    return pd.DataFrame(rows).set_index("scale")


def sensitivity(y, p, amount, term_months, margins=None, lgds=None) -> pd.DataFrame:
    """Thresholds, approval rate and profit for a grid of cost assumptions."""
    margins = config.SENSITIVITY_NET_MARGINS if margins is None else margins
    lgds = config.SENSITIVITY_LGDS if lgds is None else lgds
    rows = []
    for margin in margins:
        for lgd in lgds:
            approve = cost_rule(p, term_months, margin, lgd)
            rows.append(
                {
                    "net_margin_per_month": margin,
                    "lgd": lgd,
                    **{f"threshold_{t}m": v for t, v in thresholds_by_term(margin, lgd).items()},
                    "approval_rate": float(approve.mean()),
                    "profit_per_1000_applicants_kes": float(
                        realised_profit(y, approve, amount, term_months, margin, lgd).mean() * 1_000
                    ),
                    "profit_if_approve_all": float(
                        realised_profit(
                            y, np.ones(len(p), bool), amount, term_months, margin, lgd
                        ).mean()
                        * 1_000
                    ),
                }
            )
    return pd.DataFrame(rows)


def profit_difference(
    y,
    approve_a,
    approve_b,
    amount,
    term_months,
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    **costs,
) -> dict[str, float]:
    """Profit per 1,000 applicants of policy A minus policy B, with a 95% bootstrap interval."""
    a = realised_profit(y, approve_a, amount, term_months, **costs)
    diff = a - realised_profit(y, approve_b, amount, term_months, **costs)
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(resamples)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"difference": diff.mean() * 1_000, "low": low * 1_000, "high": high * 1_000}


def save_policy(path: Path = config.DECISION_POLICY_FILE) -> Path:
    """Record the cost assumptions and the thresholds they give, for the API (phase 4)."""
    policy = {
        "monthly_fee_rate": config.MONTHLY_FEE_RATE,
        "monthly_funding_cost": config.MONTHLY_FUNDING_COST,
        "net_margin_per_month": NET_MARGIN,
        "loss_given_default": config.LOSS_GIVEN_DEFAULT,
        "thresholds_by_term_months": thresholds_by_term(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return path


def load_policy(path: Path = config.DECISION_POLICY_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/decision_policy.py first")
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["thresholds_by_term_months"] = {
        int(k): v for k, v in policy["thresholds_by_term_months"].items()
    }
    return policy


def decide(p: float, term_months: int, policy: dict) -> str:
    """One applicant: 'approve' or 'decline', using a saved policy (what the API will call)."""
    thresholds = policy["thresholds_by_term_months"]
    if term_months not in thresholds:
        raise ValueError(
            f"No threshold for a {term_months}-month loan; terms: {sorted(thresholds)}"
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {p}")
    return "approve" if p < thresholds[term_months] else "decline"
