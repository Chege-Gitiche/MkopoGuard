"""Step 5.7: basic hardening.

1. Login rate limit. Password guessing is slowed two ways, both checked BEFORE the password:
   - per IP address: at most 20 login attempts in 5 minutes, whatever the outcome;
   - per IP address + email: at most 5 failed logins in 15 minutes (a success clears it).
   Keying failures on the pair, not the email alone, means an attacker can't lock a real
   officer out by failing on purpose from elsewhere. Over the limit: 429 with Retry-After,
   and the same message whether or not the account exists. The counts live in memory, which
   is right for one server process (Phase 8 runs one); several processes would need a shared
   store such as Redis.
2. No stack traces in responses. Any unexpected error becomes a plain 500: "Something went
   wrong on our side" plus a reference (the request id). The full traceback is logged on the
   server under the same reference, so a report from an officer can be matched to the log.
3. Validation errors (422) never echo the submitted values back, so a password can't appear in
   a response.
4. Logging: every request gets an id (X-Request-ID header and every log line); method, path,
   status and time are logged, never bodies, headers, passwords or tokens. A redacting filter
   on the log handlers masks anything that looks like a token or password anyway.
5. Security headers on every response, and "no-store" on API responses (tokens and
   predictions must not sit in a browser or proxy cache).
"""

import hashlib
import logging
import math
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("mkopoguard.api")
request_id: ContextVar[str] = ContextVar("request_id", default="-")
SERVER_ERROR = "Something went wrong on our side. Please try again; if it keeps happening, "
LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
NOT_CACHED = ("/auth", "/predict", "/predictions", "/model")  # responses with personal data


# --- 1. login rate limit -----------------------------------------------------------------------


class LoginLimiter:
    """Sliding-window counters, in memory, safe to use from several threads."""

    MAX_KEYS = 10_000  # bounded memory, even under a flood of made-up emails

    def __init__(
        self,
        attempts_per_ip: int = 20,
        ip_window_seconds: int = 300,
        failures_per_account: int = 5,
        failure_window_seconds: int = 900,
        clock=time.monotonic,
    ):
        self.attempts_per_ip, self.ip_window = attempts_per_ip, ip_window_seconds
        self.failures_per_account, self.failure_window = (
            failures_per_account,
            failure_window_seconds,
        )
        self.clock = clock
        self._attempts: dict[str, deque] = defaultdict(deque)
        self._failures: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _trim(events: deque, window: int, now: float) -> None:
        while events and events[0] <= now - window:
            events.popleft()

    def _prune(self, now: float) -> None:
        for store, window in (
            (self._attempts, self.ip_window),
            (self._failures, self.failure_window),
        ):
            if len(store) > self.MAX_KEYS:
                for key in list(store):
                    self._trim(store[key], window, now)
                    if not store[key]:
                        del store[key]

    def check(self, ip: str, email: str) -> int | None:
        """Seconds to wait if this attempt is over a limit; otherwise None (and it is counted)."""
        key = (ip, email.strip().lower())
        with self._lock:
            now = self.clock()
            self._prune(now)
            attempts, failures = self._attempts[ip], self._failures[key]
            self._trim(attempts, self.ip_window, now)
            self._trim(failures, self.failure_window, now)
            waits = []
            if len(attempts) >= self.attempts_per_ip:
                waits.append(attempts[0] + self.ip_window - now)
            if len(failures) >= self.failures_per_account:
                waits.append(failures[0] + self.failure_window - now)
            if waits:
                return max(1, math.ceil(max(waits)))
            attempts.append(now)
            return None

    def failed(self, ip: str, email: str) -> None:
        with self._lock:
            self._failures[(ip, email.strip().lower())].append(self.clock())

    def succeeded(self, ip: str, email: str) -> None:
        with self._lock:
            self._failures.pop((ip, email.strip().lower()), None)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def email_tag(email: str) -> str:
    """A short, one-way tag for an email, so logs can link attempts without storing the address."""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:10]


# --- 4. logging ----------------------------------------------------------------------------------

_SECRETS = [
    (re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]+"), "[token]"),  # a JWT
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1[token]"),
    (re.compile(r"(?i)(password[\"']?\s*[:=]\s*[\"']?)[^\"'&\s,}]+"), r"\1[hidden]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


class RequestContext(logging.Filter):
    """Adds the request id to every log line, and masks anything that looks like a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    root.setLevel(level)
    for handler in root.handlers:
        if not any(isinstance(f, RequestContext) for f in handler.filters):
            handler.addFilter(RequestContext())


# --- 2, 3, 5: errors, headers, request ids -----------------------------------------------------


def _clean_validation_errors(errors) -> list[dict]:
    """FastAPI's 422 details without the submitted values ("input") or internals."""
    return [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in errors]  # fmt: skip


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = uuid.uuid4().hex[:16]
        token = request_id.set(rid)
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # the route crashed: log everything here, tell the client nothing
            log.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = JSONResponse(
                {"detail": SERVER_ERROR + f"quote reference {rid}.", "reference": rid},
                status_code=500,
            )
        elapsed = (time.perf_counter() - started) * 1000
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
                 response.status_code, elapsed)  # fmt: skip
        response.headers["X-Request-ID"] = rid
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith(NOT_CACHED):
            response.headers["Cache-Control"] = "no-store"
        request_id.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return JSONResponse({"detail": _clean_validation_errors(error.errors())}, status_code=422)
