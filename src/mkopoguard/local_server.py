"""Step 7.4: a throwaway copy of the app, for the browser tests and the load test.

    with local_server() as server:
        server.url            # http://127.0.0.1:<free port>
        server.email, server.password

It starts the real app (Uvicorn, the packaged model, the real frontend) in its own process, on
a fresh SQLite database in a temporary folder with one officer account, and deletes it all
afterwards. The developer's own mkopoguard.db is never touched. The secret key and the
officer's password are random, made for this run only, and never printed or logged.
"""

import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mkopoguard import config

EMAIL = "officer@e2e.example"
FULL_NAME = "Test Officer"
START_SECONDS = 90  # loading the model bundle takes a few seconds; a cold Windows start longer


@dataclass(frozen=True)
class LocalServer:
    url: str
    email: str
    password: str
    log: Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_officer(url: str, password: str) -> None:
    from mkopoguard.api import db
    from mkopoguard.api.users import create_user

    db.upgrade(url)
    engine = db.make_engine(url)
    with db.make_sessions(engine)() as session:
        create_user(session, EMAIL, FULL_NAME, password, "officer")
    engine.dispose()


def _wait_until_up(url: str, process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"The app stopped while starting:\n{tail}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"The app didn't answer /health within {START_SECONDS} seconds")


@contextmanager
def local_server(log_level: str = "warning") -> Iterator[LocalServer]:
    folder = Path(tempfile.mkdtemp(prefix="mkopoguard-"))
    database_url = f"sqlite:///{(folder / 'local.db').as_posix()}"
    password = secrets.token_urlsafe(24)
    _create_officer(database_url, password)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ | {
        "MKOPOGUARD_ENVIRONMENT": "test",
        "MKOPOGUARD_SECRET_KEY": secrets.token_urlsafe(48),
        "MKOPOGUARD_DATABASE_URL": database_url,
        "MKOPOGUARD_LOG_LEVEL": "WARNING",
        # Many logins from 127.0.0.1 in a few minutes; the limit itself has its own tests
        "MKOPOGUARD_LOGIN_ATTEMPTS_PER_IP": "1000",
    }
    command = [
        sys.executable, "-m", "uvicorn", "mkopoguard.api.main:create_app", "--factory",
        "--app-dir", str(config.PROJECT_ROOT / "src"),
        "--host", "127.0.0.1", "--port", str(port), "--log-level", log_level,
    ]  # fmt: skip
    log = folder / "server.log"
    try:
        with open(log, "w", encoding="utf-8") as out:
            process = subprocess.Popen(
                command, cwd=config.PROJECT_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT
            )
            try:
                _wait_until_up(url, process, log)
                yield LocalServer(url=url, email=EMAIL, password=password, log=log)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
