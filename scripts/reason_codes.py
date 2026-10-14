"""Step 4.2: reason codes for every development applicant, checked and summarised.

Run from the project root:  python scripts/reason_codes.py
Takes about 30 seconds.

For each of the 15,910 development applicants (the test set stays closed):
- checks the promises of explain/reasons.py: exactly 3 + 2 reasons, all different, readable,
  no statement reason for a thin-file applicant, and units + other factors = the score;
- counts how often each reason appears;
- prints three worked examples (risky statement holder, thin-file, safe applicant).

Outputs: docs/results/reason_code_frequency.csv, docs/results/reason_examples.json,
         docs/img/explain_02_reason_codes.png
"""

import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkopoguard import config
from mkopoguard.bundle import load_bundle
from mkopoguard.explain import reasons as rc
from mkopoguard.models.experiment import load_training_data
from mkopoguard.viz import BLUE, ORANGE, _finish, apply_style, save

CHART = config.PROJECT_ROOT / "docs" / "img" / "explain_02_reason_codes.png"
STATEMENT_CODES = {u.code for u in rc.UNITS if u.needs_statement}
UNREADABLE = re.compile(r"\{|\}|nan|None|<NA>")


def problems(result: dict, thin: bool) -> list[str]:
    """Every promise one applicant's reasons must keep (empty list = all kept)."""
    issues = []
    raising, lowering = result["raising"], result["lowering"]
    if (len(raising), len(lowering)) != (rc.N_RAISING, rc.N_LOWERING):
        issues.append(f"{len(raising)} + {len(lowering)} reasons")
    codes = [r["code"] for r in raising + lowering]
    if len(set(codes)) != len(codes):
        issues.append("a reason appears twice")
    if thin and STATEMENT_CODES & set(codes):
        issues.append("statement reason for a thin-file applicant")
    if not thin and "no_statement" in codes:
        issues.append("'no statement' for a statement holder")
    if any(not r["text"] or UNREADABLE.search(r["text"]) for r in raising + lowering):
        issues.append("unreadable text")
    up = [r["contribution"] for r in raising]
    if up != sorted(up, reverse=True) or min(up) < max(r["contribution"] for r in lowering):
        issues.append("reasons out of order")
    return issues


def plot(freq: pd.DataFrame) -> None:
    top = freq.sort_values("total").tail(15)
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(top))
    ax.barh(y, top["raising"], color=ORANGE, height=0.7, label="Given as a reason risk is higher")
    ax.barh(
        y,
        top["lowering"],
        left=top["raising"],
        color=BLUE,
        height=0.7,
        label="Given as a reason risk is lower",
    )
    ax.set_yticks(y, top.index, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of applicants given this reason")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=8.5)
    _finish(
        ax,
        "Savings, bill worries and betting come up most often",
        "How often each reason appears in an applicant's top 3 + 2 (15,910 applicants)",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    champion = load_bundle()
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)
    reasoner = rc.Reasoner(champion.model, development)

    contribs = reasoner.explainer.contributions(development)
    units = reasoner.units(development)
    identity = float(np.abs(units.sum(axis=1) - contribs.sum(axis=1)).max())

    print(f"Explaining {len(development):,} applicants...")
    results = reasoner.explain_rows(development)
    thin = (development["has_statement"] == 0).to_numpy()
    broken = {i: p for i, r in enumerate(results) if (p := problems(r, thin[i]))}

    raising = Counter(r["code"] for res in results for r in res["raising"])
    lowering = Counter(r["code"] for res in results for r in res["lowering"])
    freq = pd.DataFrame({"raising": raising, "lowering": lowering}).fillna(0) / len(results)
    freq["total"] = freq["raising"] + freq["lowering"]
    freq = freq.sort_values("total", ascending=False)
    freq.round(4).to_csv(config.RESULTS_DIR / "reason_code_frequency.csv")
    plot(freq)

    p = np.array([r["probability_of_default"] for r in results])
    holders = np.flatnonzero(~thin)
    examples = {
        "risky_statement_holder": int(holders[np.argmax(p[holders])]),
        "thin_file": int(np.flatnonzero(thin)[0]),
        "safe_applicant": int(np.argmin(p)),
    }
    shown = {
        name: {"applicant_id": development.loc[i, "applicant_id"], **results[i]}
        for name, i in examples.items()
    }
    (config.RESULTS_DIR / "reason_examples.json").write_text(
        json.dumps(shown, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"Units + other factors = score: largest gap {identity:.1e}")
    print(f"Applicants breaking a promise: {len(broken)} of {len(results):,}")
    for i, issues in list(broken.items())[:5]:
        print(f"  {development.loc[i, 'applicant_id']}: {issues}")
    against = sum(r["direction"] == "lowers" for res in results for r in res["raising"])
    print(
        f"'Raising' slots filled by a factor that actually lowers risk "
        f"(fewer than 3 factors raise it): {against}"
    )
    material = np.mean([r["other_factors"]["material"] for r in results])
    print(f"Other factors as large as a stated reason (and 1+ point): {material:.1%} of applicants")
    print(freq.round(3).to_string())
    for name, example in shown.items():
        print(
            f"\n{name} ({example['applicant_id']}): "
            f"{example['probability_of_default']:.1%} risk "
            f"(average applicant {example['base_probability']:.1%})"
        )
        for r in example["raising"] + example["lowering"]:
            sign = "+" if r["direction"] == "raises" else "-"
            print(f"  {sign} {r['text']}  ({r['impact_points']:+.1f} points)")
        other = example["other_factors"]
        print(f"    other factors together: {other['impact_points']:+.1f} points")
    if broken or identity > 1e-9:
        raise SystemExit("Reason codes broke a promise: fix before using them")
    print(f"\nSaved reason_code_frequency.csv, reason_examples.json and {CHART.name}")


if __name__ == "__main__":
    main()
