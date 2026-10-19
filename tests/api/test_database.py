"""Step 5.2: the schema, its migrations, and the rules the database itself enforces."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from mkopoguard import config
from mkopoguard.api import db
from mkopoguard.api.main import create_app
from mkopoguard.api.models import Base, Prediction, User
from mkopoguard.api.settings import Settings

SECRET = "test-secret-key-that-is-long-enough-1234567890"


@pytest.fixture
def url(tmp_path):
    """A brand-new database, migrated to the newest schema."""
    url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    db.upgrade(url)
    return url


@pytest.fixture
def session(url):
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as s:
        yield s
    engine.dispose()


def officer(email="Asha.Wanjiru@Example.com ", role="officer") -> User:
    return User(email=email, hashed_password="not-a-real-hash", full_name="Asha Wanjiru", role=role)


def prediction(user: User, **overrides) -> Prediction:
    values = {
        "officer": user,
        "applicant_ref": "APP-2026-0001",
        "inputs": {"age": 31, "term_months": 3, "statement": {"rows": 412}},
        "probability": 0.183,
        "band": "medium",
        "decision": "approve",
        "reasons": {"raising": [{"code": "betting"}], "lowering": [{"code": "saved_past_year"}]},
        "model_version": config.MODEL_VERSION,
    }
    return Prediction(**values | overrides)


# --- migrations -------------------------------------------------------------------------------


def test_migration_creates_both_tables_with_every_column(url):
    inspector = inspect(db.make_engine(url))
    assert {"users", "predictions"} <= set(inspector.get_table_names())
    users = {c["name"] for c in inspector.get_columns("users")}
    predictions = {c["name"] for c in inspector.get_columns("predictions")}
    assert users == {
        "id", "email", "hashed_password", "full_name", "role", "is_active", "created_at"
    }  # fmt: skip
    assert predictions == {
        "id", "officer_id", "applicant_ref", "inputs", "probability", "band", "decision",
        "reasons", "model_version", "created_at", "threshold",
    }  # fmt: skip


def test_models_and_migrations_agree(url):
    """Nothing in api/models.py is missing from the migrations (and nothing extra)."""
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    engine = db.make_engine(url)
    with engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert diff == []


def test_migrations_go_down_and_up_again(url):
    db.downgrade(url, "base")
    engine = db.make_engine(url)
    assert not {"users", "predictions"} & set(inspect(engine).get_table_names())
    db.upgrade(url)
    assert db.current_revision(engine) == db.head_revision()


def test_schema_check_accepts_the_newest_and_refuses_an_unmigrated_database(url, tmp_path):
    assert db.check_schema(db.make_engine(url)) == db.head_revision()
    empty = db.make_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    with pytest.raises(db.SchemaOutOfDate, match="alembic upgrade head"):
        db.check_schema(empty)


# --- rules the database enforces ------------------------------------------------------------------


def test_a_user_is_saved_with_defaults_and_a_normalised_email(session):
    session.add(officer())
    session.commit()
    user = session.scalars(select(User)).one()
    assert user.email == "asha.wanjiru@example.com"
    assert user.role == "officer" and user.is_active is True
    assert user.created_at is not None


def test_one_account_per_email_however_it_is_typed(session):
    session.add(officer("asha@example.com"))
    session.commit()
    session.add(officer("  ASHA@example.com"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_only_known_roles_are_allowed(session):
    session.add(officer(role="superuser"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_a_prediction_keeps_its_inputs_reasons_and_model_version(session):
    user = officer()
    session.add(prediction(user))
    session.commit()
    saved = session.scalars(select(Prediction)).one()
    assert saved.inputs["statement"] == {"rows": 412}
    assert saved.reasons["raising"][0]["code"] == "betting"
    assert saved.model_version == config.MODEL_VERSION
    assert saved.officer.email == user.email and saved.created_at is not None


def test_a_prediction_must_belong_to_an_existing_officer(session):
    session.add(
        Prediction(
            officer_id=999,
            applicant_ref="APP-X",
            inputs={},
            probability=0.1,
            band="low",
            decision="approve",
            reasons={},
            model_version="1.1.0",
        )
    )
    with pytest.raises(IntegrityError):  # foreign keys are on, even in SQLite
        session.commit()


@pytest.mark.parametrize("overrides", [{"probability": 1.5}, {"probability": -0.1},
                                       {"decision": "maybe"}])  # fmt: skip
def test_impossible_results_are_refused(session, overrides):
    session.add(prediction(officer(), **overrides))
    with pytest.raises(IntegrityError):
        session.commit()


def test_an_officer_with_history_cannot_be_deleted(session):
    user = officer()
    session.add(prediction(user))
    session.commit()
    session.delete(user)
    with pytest.raises(IntegrityError):  # deactivate instead: history keeps its officer
        session.commit()


def test_history_comes_newest_first(session):
    from datetime import UTC, datetime, timedelta

    user = officer()
    start = datetime(2026, 10, 1, tzinfo=UTC)
    for day in (1, 3, 2):
        session.add(prediction(user, applicant_ref=f"APP-{day}", created_at=start + timedelta(day)))
    session.commit()
    session.expire_all()
    assert [p.applicant_ref for p in user.predictions] == ["APP-3", "APP-2", "APP-1"]


def test_repr_never_shows_the_password_hash():
    assert "not-a-real-hash" not in repr(officer())


# --- the app and the database ---------------------------------------------------------------------


def test_app_refuses_to_start_on_an_unmigrated_database(tmp_path):
    empty = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=empty)
    with pytest.raises(db.SchemaOutOfDate, match="alembic upgrade head"):
        with TestClient(create_app(settings)):
            pass


def test_health_reports_the_database(url):
    settings = Settings(_env_file=None, secret_key=SECRET, database_url=url)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["database"] == "ok"


def test_migrations_keep_existing_history(tmp_path):
    """Regression (step 5.3): rebuilding `users` on SQLite must not fail, or lose rows, while
    predictions point at it."""
    url = f"sqlite:///{(tmp_path / 'history.db').as_posix()}"
    db.upgrade(url, "0001")
    engine = db.make_engine(url)
    with engine.begin() as connection:  # rows as they were at 0001: plain SQL, not today's model
        connection.exec_driver_sql(
            "INSERT INTO users (id, email, hashed_password, full_name, role, is_active, "
            "created_at) VALUES (1, 'asha.wanjiru@example.com', 'h', 'Asha', 'officer', 1, "
            "'2026-10-01 09:00:00')"
        )
        connection.exec_driver_sql(
            "INSERT INTO predictions (officer_id, applicant_ref, inputs, probability, band, "
            "decision, reasons, model_version, created_at) VALUES (1, 'APP-1', '{}', 0.2, "
            "'high', 'decline', '{}', '1.0.0', '2026-10-01 09:00:00')"
        )
    db.upgrade(url)
    db.downgrade(url, "0001")
    db.upgrade(url)
    with db.make_sessions(engine)() as session:
        saved = session.scalars(select(Prediction)).one()
        assert saved.officer.email == "asha.wanjiru@example.com"
        assert saved.threshold is None  # made before 0003: no threshold was kept
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_times_are_always_returned_in_utc(session):
    from datetime import UTC, datetime, timedelta, timezone

    nairobi = timezone(timedelta(hours=3))
    session.add(prediction(officer(), created_at=datetime(2026, 10, 14, 13, 0, tzinfo=nairobi)))
    session.commit()
    session.expire_all()
    saved = session.scalars(select(Prediction)).one()
    assert saved.created_at == datetime(2026, 10, 14, 10, 0, tzinfo=UTC)
    assert saved.created_at.utcoffset() == timedelta(0)
