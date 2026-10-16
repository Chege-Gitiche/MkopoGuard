"""Step 5.1: the FastAPI app.

- Loads the model bundle ONCE at startup (checksums, golden examples and the explainer are
  checked by load_bundle). If the bundle is broken the app refuses to start, so the host's
  health check fails loudly instead of the first applicant getting an error.
- GET /health: is the app up, and which model is it serving? Public, for the host's uptime
  check (Render pings it, Phase 8). It reveals nothing secret.
- Serves the frontend/ folder at "/" from the same origin, so the browser needs no CORS rules.
  API routes are registered first and always win over a file with the same path.

Run locally:  uvicorn --app-dir src mkopoguard.api.main:create_app --factory --reload
then open http://127.0.0.1:8000 (the app) or http://127.0.0.1:8000/docs (the API docs).
"""

import logging
import mimetypes
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mkopoguard import config
from mkopoguard.api import auth, hardening, history, openapi, predict
from mkopoguard.api.db import check_schema, database_ok, make_engine, make_sessions
from mkopoguard.api.settings import Settings, get_settings
from mkopoguard.bundle import load_bundle

log = logging.getLogger("mkopoguard.api")

# The frontend's file types, set here rather than guessed. On Windows, Python reads file types
# from the registry, which can be missing or wrong (.woff2 often is), and the browser is told
# "nosniff", so it must get the right type.
STATIC_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".txt": "text/plain",
}
for _ext, _kind in STATIC_TYPES.items():
    mimetypes.add_type(_kind, _ext)


class Health(BaseModel):
    status: str
    model_name: str
    model_version: str
    database: str
    uptime_seconds: float


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings; uvicorn calls it with none (--factory)."""
    settings = settings or get_settings()
    hardening.configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle = load_bundle(settings.bundle_dir)  # raises if corrupted: the app won't start
        engine = make_engine(settings.database_url)
        revision = check_schema(engine)  # raises if `alembic upgrade head` wasn't run
        app.state.bundle, app.state.engine = bundle, engine
        app.state.sessions = make_sessions(engine)
        app.state.reasoner = bundle.reasoner()  # reason codes from the bundle alone (v1.1.0)
        app.state.started = time.monotonic()
        log.info(
            "Serving %s v%s (%s)",
            bundle.manifest["name"],
            bundle.manifest["version"],
            settings.environment,
        )
        log.info("Database schema at migration %s", revision)
        yield
        engine.dispose()

    app = FastAPI(
        title="MkopoGuard API",
        version=config.MODEL_VERSION,
        description=openapi.DESCRIPTION,
        openapi_tags=openapi.TAGS,
        swagger_ui_parameters={"tryItOutEnabled": True, "displayRequestDuration": True},
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.login_limiter = hardening.LoginLimiter(
        settings.login_attempts_per_ip,
        settings.login_ip_window_seconds,
        settings.login_failures_per_account,
        settings.login_failure_window_seconds,
    )
    hardening.install(app)  # request ids, no stack traces in responses, security headers

    @app.get("/health", response_model=Health, tags=["system"], summary="Health check")
    def health(request: Request) -> Health:
        """Uptime check: the app is running and has a valid model loaded."""
        bundle = request.app.state.bundle
        return Health(
            status="ok",
            model_name=bundle.manifest["name"],
            model_version=bundle.manifest["version"],
            database="ok" if database_ok(request.app.state.engine) else "unavailable",
            uptime_seconds=round(time.monotonic() - request.app.state.started, 1),
        )

    app.include_router(auth.router)
    app.include_router(predict.router)
    app.include_router(history.router)

    # Last, so every API route above takes precedence over a file of the same name
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app
