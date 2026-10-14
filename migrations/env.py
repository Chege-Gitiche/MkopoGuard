"""Alembic environment: where to connect, and which models describe the schema.

The URL comes from the caller (tests and scripts set sqlalchemy.url) or else from the app's own
settings (MKOPOGUARD_DATABASE_URL or .env), so there is one place to configure it.
"""

from logging.config import fileConfig

from alembic import context

from mkopoguard.api.db import make_engine
from mkopoguard.api.models import Base
from mkopoguard.api.settings import DatabaseSettings

config = context.config
if config.config_file_name and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata
url = config.get_main_option("sqlalchemy.url") or DatabaseSettings().database_url


def run_migrations_offline() -> None:
    """Write the SQL to the screen instead of running it (alembic upgrade head --sql)."""
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite can't ALTER most things; batch mode copies the table
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
