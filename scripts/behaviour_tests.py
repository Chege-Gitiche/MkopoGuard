"""Step 4.3: behaviour (directional) tests of the champion on every development applicant.

Run from the project root:  python scripts/behaviour_tests.py
Takes about 10 seconds.

For each check in explain/behaviour.py: change one thing (or a few linked things) about the
applicants it applies to, keep everything else equal, and measure how the model's probability
of default moves. The test set stays closed: these are the 15,910 development applicants.

Outputs: docs/results/behaviour_tests.csv, docs/img/explain_03_behaviour.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from mkopoguard import config
from mkopoguard.bundle import load_bundle
from mkopoguard.explain import behaviour as bh
from mkopoguard.models import decision as dec
from mkopoguard.models.experiment import load_training_data
from mkopoguard.viz import BLUE, GREY, ORANGE, _finish, apply_style, save

CHART = config.PROJECT_ROOT / "docs" / "img" / "explain_03_behaviour.png"
OUT = config.RESULTS_DIR / "behaviour_tests.csv"


def plot(results: pd.DataFrame) -> None:
    checks = results[results["expect"] != bh.NONE]
    n_pass, n_checks = int(checks["passed"].sum()), len(checks)
    shown = results.sort_values("mean_change_pts")
    colours = [
        GREY if r.expect == bh.NONE else ORANGE if r.passed is False else BLUE
        for r in shown.itertuples()
    ]
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 10))
    y = np.arange(len(shown))
    ax.barh(y, shown["mean_change_pts"], color=colours, height=0.7)
    ax.set_yticks(y, shown["description"], fontsize=7.5)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average change in probability of default (percentage points)")
    handles = [
        Patch(color=BLUE, label="Moves risk the expected way for everyone"),
        Patch(color=ORANGE, label="Moves risk the wrong way for some (known flip)"),
        Patch(color=GREY, label="No expectation (measured only)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    _finish(
        ax,
        f"{n_pass} of {n_checks} behaviour checks pass; the failures involve overdraft count "
        "and median inflow",
        "One change at a time, everything else equal (15,910 development applicants)",
        "none",
    )
    save(fig, CHART)


def main() -> None:
    champion = load_bundle()
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)

    def decide(frame, p):
        terms = frame["term_months"].astype(int)
        return np.array(
            [dec.decide(pi, t, champion.policy) for pi, t in zip(p, terms, strict=True)]
        )

    results = bh.run_all(champion.predict_proba, development, decide=decide)
    results.round(4).to_csv(OUT)
    plot(results)

    checks = results[results["expect"] != bh.NONE]
    failed = checks[checks["passed"] == False]  # noqa: E712 (None means "not a test")
    unexpected = sorted(set(failed.index) - set(bh.KNOWN_FLIPS))
    fixed = sorted(set(bh.KNOWN_FLIPS) - set(failed.index))
    print(f"Behaviour checks: {int(checks['passed'].sum())} of {len(checks)} pass")
    cols = ["n", "mean_change_pts", "share_wrong", "worst_wrong_pts", "share_decisions_wrong"]
    print(checks[cols + ["passed"]].round(3).to_string())
    print("\nReports (no expectation):")
    reports = results[results["expect"] == bh.NONE]
    print(
        reports[["n", "mean_change_pts", "min_change_pts", "max_change_pts"]].round(3).to_string()
    )
    print(f"\nKnown flips still failing: {sorted(set(failed.index) & set(bh.KNOWN_FLIPS))}")
    if unexpected:
        print(f"NEW FAILURES (not known flips): {unexpected}")
    if fixed:
        print(f"Known flips that now pass (update KNOWN_FLIPS): {fixed}")
    print(
        f"Saved {OUT.relative_to(config.PROJECT_ROOT)} and {CHART.relative_to(config.PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
