"""Step 5.2: database connections and migrations.

- make_engine() turns on foreign keys for SQLite (off by default there), so the rules in the
  schema are enforced locally exactly as on Postgres in production.
- The app checks at startup that the database is at the newest migration. If someone forgot
  `alembic upgrade head`, it refuses to start with a message saying so, instead of failing on
  the first login with "no such table".
"""

from collections.abc import Iterator

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi import Request
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from mkopoguard import config

ALEMBIC_INI = config.PROJECT_ROOT / "alembic.ini"


class SchemaOutOfDate(RuntimeError):
    """The database isn't at the newest migration."""


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _foreign_keys_on(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True)  # reconnects after the host drops idle links


def make_sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


# --- migrations ----------------------------------------------------------------------------------


def alembic_config(url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(config.PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # % is special in .ini files
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


def head_revision() -> str:
    return ScriptDirectory.from_config(alembic_config("sqlite://")).get_current_head()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def check_schema(engine: Engine) -> str:
    """The database's revision, if it is the newest one; otherwise SchemaOutOfDate."""
    current, head = current_revision(engine), head_revision()
    if current != head:
        raise SchemaOutOfDate(
            f"Database is at migration {current or 'none'}, the code needs {head}. "
            "Run: alembic upgrade head"
        )
    return current


def database_ok(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # health checks report, they don't crash
        return False


# --- per-request session (used by the routes from step 5.3) ------------------------------------


def get_session(request: Request) -> Iterator[Session]:
    """One session per request, always closed; commit explicitly in the route."""
    with request.app.state.sessions() as session:
        yield session
