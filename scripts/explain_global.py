"""Step 4.1: global explanations for the packaged champion, checked against the simulator.

Run from the project root:  python scripts/explain_global.py
Takes a few seconds.

Explains the bundle in models/champion/ (exactly the model the API will use) on the 15,910
development applicants (training + validation). The test set stays closed: explaining a model
describes the model, so it doesn't need never-seen applicants.

Checks (a mismatch means a bug somewhere):
1. Additivity: base value + contributions = the model's log-odds, for every applicant.
2. Agreement with the shap library's LinearExplainer (if shap is installed).
3. Known answer: the most important inputs must be the simulator's real drivers, and the
   non-drivers together must carry only a small share of the importance.

Outputs: docs/results/shap_importance.csv, docs/results/shap_groups.csv,
         docs/img/explain_01_global.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from mkopoguard import config
from mkopoguard.bundle import load_bundle
from mkopoguard.explain import design
from mkopoguard.explain.linear_shap import LinearShap, combine, importance
from mkopoguard.features.groups import FEATURE_GROUPS
from mkopoguard.models.experiment import load_training_data
from mkopoguard.viz import BLUE, GREY, ORANGE, TEXT_MUTED, _finish, apply_style, save

CHART = config.PROJECT_ROOT / "docs" / "img" / "explain_01_global.png"
GROUP_OF = {col: group for group, cols in FEATURE_GROUPS.items() for col in cols}


def importance_table(contribs: pd.DataFrame, thin: np.ndarray) -> pd.DataFrame:
    imp = importance(contribs)
    table = pd.DataFrame(
        {
            "group": [GROUP_OF[c] for c in imp.index],
            "driver": [design.is_driver(c) for c in imp.index],
            "design_role": [design.DRIVERS.get(c, "not used by the simulator") for c in imp.index],
            "mean_abs_shap": imp,
            "share": imp / imp.sum(),
            "mean_abs_statement_holders": contribs.loc[~thin, imp.index].abs().mean(),
            "mean_abs_thin_file": contribs.loc[thin, imp.index].abs().mean(),
        }
    )
    table.loc[design.STATEMENT_FLAG, "design_role"] = (
        "statement flag: carries the thin-file difference (see note)"
    )
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def shap_library_gap(explainer: LinearShap, model, background, sample) -> float | None:
    """Largest difference from shap.LinearExplainer, or None if shap isn't installed."""
    try:
        import shap
    except ImportError:
        return None
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    bg = prep.transform(background[explainer.columns]).to_numpy()
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    theirs = shap.LinearExplainer(lr, masker).shap_values(
        prep.transform(sample[explainer.columns]).to_numpy()
    )
    ours = explainer.transformed_contributions(sample).to_numpy()
    return float(np.abs(theirs - ours).max())


def plot(table: pd.DataFrame, contribs: pd.DataFrame, values: pd.DataFrame) -> None:
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 8.2), width_ratios=[1, 1.15])

    order = table.index[::-1]
    colours = [
        ORANGE if c == design.STATEMENT_FLAG else BLUE if table.loc[c, "driver"] else GREY
        for c in order
    ]
    left.barh(range(len(order)), table.loc[order, "mean_abs_shap"], color=colours, height=0.7)
    left.set_yticks(range(len(order)), order, fontsize=8)
    left.grid(axis="y", visible=False)
    left.set_xlabel("Average size of contribution (log-odds)")
    legend = [("Simulator driver", BLUE), ("Not a driver", GREY), ("Statement flag", ORANGE)]
    handles = [Patch(color=colour, label=label) for label, colour in legend]
    left.legend(handles=handles, loc="lower right", fontsize=8.5)
    _finish(
        left,
        "The model leans on the simulator's real drivers",
        "Mean |SHAP| per input, 15,910 development applicants",
        "none",
    )

    top = [c for c in table.index if c != design.STATEMENT_FLAG][:12][::-1]
    cmap = LinearSegmentedColormap.from_list("lowhigh", [BLUE, "#e6e5e0", ORANGE])
    rng = np.random.default_rng(0)
    sample = contribs.sample(min(2_000, len(contribs)), random_state=0).index
    for i, col in enumerate(top):
        x = contribs.loc[sample, col].to_numpy()
        raw = pd.to_numeric(values.loc[sample, col], errors="coerce")
        if raw.isna().all():  # categorical: colour by its own contribution instead
            raw = pd.Series(x, index=sample)
        shade = raw.rank(pct=True).fillna(0.5).to_numpy()
        right.scatter(
            x, i + rng.uniform(-0.3, 0.3, len(x)), c=shade, cmap=cmap, s=4, alpha=0.6, lw=0
        )
    right.axvline(0, color=TEXT_MUTED, lw=0.8)
    right.set_yticks(range(len(top)), top, fontsize=8)
    right.grid(axis="y", visible=False)
    right.set_xlabel("Contribution to risk (log-odds): left = safer, right = riskier")
    _finish(
        right,
        "High betting and no savings push risk up",
        "One dot per applicant (2,000 shown); colour: low (blue) to high (orange) value",
        "none",
    )
    fig.tight_layout()
    save(fig, CHART)


def main() -> None:
    champion = load_bundle()  # checksums + golden examples
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)
    thin = (development["has_statement"] == 0).to_numpy()

    explainer = LinearShap(champion.model, development)
    contribs = explainer.contributions(development)

    model_log_odds = champion.model.decision_function(development[champion.columns])
    additivity = float(np.abs(explainer.log_odds(contribs) - model_log_odds).max())
    library_gap = shap_library_gap(explainer, champion.model, development, validation.head(500))

    table = importance_table(contribs, thin)
    statement_block = {"statement": FEATURE_GROUPS["statement"]}
    groups = importance(combine(contribs, {g: c for g, c in FEATURE_GROUPS.items()}))
    with_block = importance(combine(contribs, statement_block))
    check = design.check_against_design(table["mean_abs_shap"])

    table.round(4).to_csv(config.RESULTS_DIR / "shap_importance.csv")
    groups.round(4).rename("mean_abs_shap").to_csv(config.RESULTS_DIR / "shap_groups.csv")
    plot(table, contribs, development[champion.columns])

    print(f"Base value (average log-odds): {explainer.base_value:.3f}")
    print(f"1. Additivity: largest gap from the model's log-odds {additivity:.1e}")
    if library_gap is None:
        print("2. shap library not installed: comparison skipped")
    else:
        print(f"2. shap.LinearExplainer: largest difference {library_gap:.1e}")
    show = ["rank", "group", "driver", "mean_abs_shap", "share", "mean_abs_thin_file"]
    print(table[show].round(3).to_string())
    print("By feature group:")
    print(groups.round(3).to_string())
    print(f"With the statement as one block: {with_block.round(3).head(5).to_dict()}")
    block = combine(contribs, statement_block)["statement"]
    print(
        f"Thin-file applicants' statement block: mean {block[thin].mean():+.3f}, "
        f"spread {block[thin].std():.1e} (constant: there is no statement to explain)"
    )
    ok = check["passed"]
    print(f"3. Known answer: {'PASS' if ok else 'FAIL'}")
    print(
        f"   top {design.TOP_DRIVERS} inputs (statement flag aside) are all drivers: "
        f"{check['top_inputs_all_drivers']} {check['top_non_drivers'] or ''}"
    )
    print(
        f"   non-drivers' share of importance: {check['non_driver_share']:.1%} "
        f"(limit {design.MAX_NON_DRIVER_SHARE:.0%})"
    )
    print(f"   drivers the model barely uses: {check['under_used_drivers']}")
    if not ok or additivity > 1e-9 or (library_gap is not None and library_gap > 1e-9):
        raise SystemExit("A check failed: investigate before trusting the explanations")
    print(f"Saved shap_importance.csv, shap_groups.csv and {CHART.name}")


if __name__ == "__main__":
    main()
