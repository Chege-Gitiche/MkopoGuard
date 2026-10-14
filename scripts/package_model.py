"""Step 3.13: package the champion tested in 3.12 into a self-contained bundle.

Run from the project root:  python scripts/package_model.py        (add --force to rebuild)

1. Find the champion_final run in MLflow and check its frozen decisions match the current
   files (if a decision changed after the test, the tested model is no longer the one the
   files describe, and packaging is refused).
2. Load that exact fitted pipeline: it is not retrained here.
3. Write models/champion/: model, input schema, decision policy, golden examples, model card
   and a manifest with checksums and library versions.
4. Reload the bundle from disk, check the checksums, replay the golden examples, and check
   it gives the same probabilities as the MLflow model on every validation applicant.

The bundle is small (a logistic regression) and is committed to Git, so the API and CI can
use it without MLflow.
"""

import argparse
import ast
import hashlib
import json
import shutil

import mlflow
import numpy as np
import pandas as pd

from mkopoguard import bundle, config
from mkopoguard.explain.reasons import Reasoner
from mkopoguard.features.columns import TARGET
from mkopoguard.model_card import build_model_card
from mkopoguard.models.experiment import load_training_data
from mkopoguard.models.final import FROZEN_FILES
from mkopoguard.tracking import load_pipeline, setup_tracking


def champion_run():
    """The latest champion_final run, which must match the current frozen decisions."""
    runs = mlflow.search_runs(
        experiment_names=[config.MLFLOW_EXPERIMENT],
        filter_string="tags.mlflow.runName = 'champion_final' AND tags.evaluated_on = 'test'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise SystemExit("No champion_final run in MLflow; run scripts/final_test.py (3.12) first")
    run = runs.iloc[0]
    changed = changed_decisions(ast.literal_eval(run["tags.frozen"]))
    if changed:
        raise SystemExit(
            f"Decision file(s) {changed} changed after the final test, so the tested model no "
            "longer matches them. Undo the change: the test can't be re-opened to refresh it."
        )
    return run


def changed_decisions(tested: dict[str, str]) -> list[str]:
    """Decision files whose content differs from when the test set was opened.

    Line endings don't count: Git on Windows may turn "\n" into "\r\n" (or back) on checkout,
    which changes a file's bytes but not the decision in it.
    """
    changed = []
    for name, path in FROZEN_FILES.items():
        raw = path.read_bytes()
        unix = raw.replace(b"\r\n", b"\n")
        variants = {raw, unix, unix.replace(b"\n", b"\r\n")}
        prints = {hashlib.md5(v).hexdigest()[:12] for v in variants}
        if tested.get(name) not in prints:
            changed.append(name)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the champion into models/champion/.")
    parser.add_argument("--force", action="store_true", help="replace an existing bundle")
    args = parser.parse_args()

    setup_tracking()
    run = champion_run()
    run_id = run["run_id"]
    print(f"Champion run {run_id} (commit {run['tags.git_commit']})")
    pipeline = load_pipeline(run_id)

    report = json.loads(config.TEST_REPORT_FILE.read_text(encoding="utf-8"))
    selection = json.loads(config.FEATURE_SELECTION_FILE.read_text(encoding="utf-8"))
    policy = json.loads(config.DECISION_POLICY_FILE.read_text(encoding="utf-8"))
    policy["thresholds_by_term_months"] = {
        int(k): v for k, v in policy["thresholds_by_term_months"].items()
    }

    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)

    out = config.BUNDLE_DIR
    if (out / bundle.MANIFEST).exists() and not args.force:
        raise SystemExit(f"{out} already exists; use --force to rebuild it")

    card = build_model_card(report, policy, selection["kept_groups"], run_id)
    provenance = {
        "mlflow_run_id": run_id,
        "git_commit_of_test": run["tags.git_commit"],
        "frozen_decisions": report["frozen_decisions"],
        "test_pr_auc": report["ranking"]["pr_auc"]["value"],
        "trained_on": report["trained_on"],
    }
    # Build in a temporary folder, so a failure never leaves a half-written bundle behind.
    # Golden examples come from validation applicants: the test set stays closed.
    staging = out.with_name(out.name + "_staging")
    shutil.rmtree(staging, ignore_errors=True)
    bundle.build_bundle(pipeline, development, validation, staging, policy, card, provenance)

    # Check: reload from disk only, then compare against the MLflow model
    loaded = bundle.load_bundle(staging)  # checksums + golden examples
    p_bundle = loaded.predict_proba(validation)
    p_mlflow = pipeline.predict_proba(validation[loaded.columns])[:, 1]
    gap = float(np.max(np.abs(p_bundle - p_mlflow)))
    if gap > config.GOLDEN_TOLERANCE:
        raise SystemExit(f"Bundle predictions differ from the MLflow model by {gap:.3g}")
    # Reasons from the saved background must be exactly the reasons from the full data
    sample = validation.sample(500, random_state=0)
    if loaded.reasoner().explain_rows(sample) != Reasoner(pipeline, development).explain_rows(
        sample
    ):
        raise SystemExit("Reasons from the bundle differ from reasons built from the data")
    problems = bundle.check_input(validation, loaded.schema)
    if problems["errors"]:
        raise SystemExit(f"Validation applicants fail the input schema: {problems['errors']}")

    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    config.MODEL_CARD_FILE.write_text(card, encoding="utf-8", newline="\n")  # same text as bundle
    sizes = {f.name: f.stat().st_size for f in out.iterdir()}
    print(f"Bundle {config.MODEL_NAME} v{config.MODEL_VERSION} written to {out}")
    for name, size in sorted(sizes.items()):
        print(f"  {name:<26} {size / 1024:7.1f} KB")
    print(f"  {len(loaded.columns)} input columns, {len(loaded.golden)} golden examples")
    print(
        f"Reloaded from disk: checksums OK, golden examples OK, max difference from the "
        f"MLflow model on {len(validation):,} validation applicants: {gap:.1e}"
    )
    default_rate = validation[TARGET].mean()
    print(f"Validation check: mean predicted {p_bundle.mean():.3f} vs actual {default_rate:.3f}")
    print("Reason codes from the bundle match reasons built from the data (500 applicants)")
    print(f"Model card written to {config.MODEL_CARD_FILE.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
