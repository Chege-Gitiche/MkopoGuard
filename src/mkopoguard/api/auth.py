"""Step 5.3: login (POST /auth/login), who am I (GET /auth/me), and the guard on every
officer route (get_current_user).

Every failure to log in gets the same answer, "Incorrect email or password", whether the email
is unknown, the password is wrong or the account is deactivated, so the login form can't be
used to find out who has an account.

Roles
  officer  scores applicants; sees only their own history
  admin    the same, plus admin tasks (later)
  demo     the public portfolio account: can score sample applicants and read its own sample
           history, but never writes anything to the database (enforced in step 5.5)
"""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from mkopoguard.api import security
from mkopoguard.api.db import get_session
from mkopoguard.api.hardening import client_ip, email_tag
from mkopoguard.api.models import User
from mkopoguard.api.users import find_user

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("mkopoguard.api.auth")
SessionDep = Annotated[Session, Depends(get_session)]
# tokenUrl makes the "Authorize" button in /docs log in through /auth/login
bearer = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

LOGIN_FAILED = "Incorrect email or password"
NOT_AUTHENTICATED = "Not logged in, or the session has expired"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserOut(BaseModel):
    """What the API shows about a user: never the password hash."""

    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


def _unauthorized(detail: str = NOT_AUTHENTICATED) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_current_user(
    request: Request,
    token: Annotated[str | None, Depends(bearer)],
    session: SessionDep,
) -> User:
    """The logged-in, active user, or 401. Put this on every route an officer uses."""
    if not token:
        raise _unauthorized()
    secret = request.app.state.settings.secret_key.get_secret_value()
    try:
        claims = security.decode_access_token(token, secret)
        user_id = int(claims["sub"])
    except (security.InvalidToken, ValueError):
        raise _unauthorized() from None
    user = session.get(User, user_id)
    if user is None or not user.is_active:  # deactivated since the token was issued
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_roles(*roles: str):
    """Dependency: the current user must have one of these roles, otherwise 403."""

    def check(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed for this account")
        return user

    return check


@router.post("/login", response_model=Token)
def login(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    """Email (in the `username` field) and password in; a short-lived access token out."""
    settings = request.app.state.settings
    limiter, ip, tag = request.app.state.login_limiter, client_ip(request), email_tag(form.username)
    wait = limiter.check(ip, form.username)  # before any password work
    if wait is not None:
        log.warning("Login rate-limited: ip=%s account=%s wait=%ss", ip, tag, wait)
        minutes = max(1, -(-wait // 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Wait {minutes} minute(s) and try again.",
            headers={"Retry-After": str(wait)},
        )
    user = find_user(session, form.username)
    hashed = user.hashed_password if user else security.dummy_hash()
    password_ok = security.verify_password(form.password, hashed)  # always run: same timing
    if user is None or not password_ok or not user.is_active:
        limiter.failed(ip, form.username)
        log.info("Login failed: ip=%s account=%s", ip, tag)  # never the email or password
        raise _unauthorized(LOGIN_FAILED)
    limiter.succeeded(ip, form.username)
    log.info("Login ok: user=%s", user.id)
    token = security.create_access_token(
        user.id,
        user.role,
        settings.secret_key.get_secret_value(),
        settings.access_token_minutes,
    )
    return Token(access_token=token, expires_in=settings.access_token_minutes * 60)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    """The logged-in user's own details."""
    return user
