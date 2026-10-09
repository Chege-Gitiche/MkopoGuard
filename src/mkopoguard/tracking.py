"""Step 3.1: experiment tracking with MLflow.

Every model run in Phase 3 goes through start_run(), which records:
- what you set: parameters, metrics and the fitted pipeline (logged by the caller)
- what you might forget: the Git commit, a fingerprint of the split file and of the model
  table, and which split the metrics were measured on

Runs are stored locally in mlruns/ (a SQLite database plus an artifacts folder), which is
git-ignored. View them with:  python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

import os

# MLflow sends anonymous usage telemetry by default; a credit-risk project shouldn't send
# anything out unasked. Must be set before mlflow is imported.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import hashlib  # noqa: E402
import subprocess  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path, PurePath  # noqa: E402

import mlflow  # noqa: E402

from mkopoguard import config  # noqa: E402
from mkopoguard.features.preprocess import SKOPS_TRUSTED_TYPES  # noqa: E402

DB_FILE = "mlflow.db"
ARTIFACTS_DIR = "artifacts"
SPLITS_ALLOWED = ("cv", "train", "validation", "test")


def tracking_uri(root: PurePath = config.MLRUNS_DIR) -> str:
    """SQLite URI for the tracking database. as_posix() keeps Windows paths valid."""
    return f"sqlite:///{(root / DB_FILE).as_posix()}"


def setup_tracking(
    experiment: str = config.MLFLOW_EXPERIMENT, root: Path = config.MLRUNS_DIR
) -> str:
    """Point MLflow at the project's local store and select the experiment. Returns its ID."""
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(root))
    found = mlflow.get_experiment_by_name(experiment)
    if found is None:
        artifact_location = (root / ARTIFACTS_DIR).resolve().as_uri()
        experiment_id = mlflow.create_experiment(experiment, artifact_location=artifact_location)
    else:
        experiment_id = found.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def file_fingerprint(path: Path) -> str:
    """Short MD5 of a file, or 'missing' if it doesn't exist."""
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_commit() -> str:
    """The current Git commit, with '-dirty' if tracked files have uncommitted changes."""
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def provenance_tags() -> dict[str, str]:
    """Tags that let any run be traced back to the exact code and data that produced it."""
    return {
        "git_commit": git_commit(),
        "split_file_md5": file_fingerprint(config.SPLIT_FILE),
        "features_file_md5": file_fingerprint(config.DATA_PROCESSED / "features.parquet"),
        "split_seed": str(config.SPLIT_SEED),
        "simulation_seed": str(config.SIMULATION_SEED),
    }


@contextmanager
def start_run(name: str, evaluated_on: str, step: str, **tags):
    """Open an MLflow run with provenance tags. Use it as:  with start_run(...) as run:

    evaluated_on says where the logged metrics come from ('cv', 'train', 'validation' or
    'test'), so a test-set score can never be mistaken for a validation score.
    Call setup_tracking() once before the first run.
    """
    if evaluated_on not in SPLITS_ALLOWED:
        raise ValueError(f"evaluated_on must be one of {SPLITS_ALLOWED}, not {evaluated_on!r}")
    all_tags = {**provenance_tags(), "evaluated_on": evaluated_on, "guide_step": step}
    all_tags.update({k: str(v) for k, v in tags.items()})
    with mlflow.start_run(run_name=name, tags=all_tags) as run:
        yield run


def log_pipeline(pipeline, name: str = "model"):
    """Save a fitted pipeline in MLflow's safe skops format, trusting only our own types."""
    return mlflow.sklearn.log_model(pipeline, name=name, skops_trusted_types=SKOPS_TRUSTED_TYPES)


def load_pipeline(run_id: str, name: str = "model"):
    """Load a pipeline saved by log_pipeline()."""
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{name}")
