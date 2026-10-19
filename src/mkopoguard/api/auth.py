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

Two ways to send the login (step 6.3)
  Bearer header   for /docs, scripts and tests: the token comes back in the JSON body and is
                  sent as "Authorization: Bearer <token>".
  Session cookie  for the web app: when the login request carries "X-Requested-With: fetch"
                  (which only the app's own JavaScript sends), the token goes into an HttpOnly,
                  SameSite=Strict cookie instead, and the body holds no token. JavaScript can't
                  read the cookie, so a script injected into a page can't steal the login.
  A request that changes something (POST, PUT, PATCH, DELETE) and is logged in by the cookie
  must also carry "X-Requested-With: fetch". Another website can't add that header to a
  request it makes the browser send, so it can't act as the officer (cross-site request
  forgery). SameSite=Strict already stops the cookie being sent from other sites; this is a
  second lock on the same door.
"""

import logging
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from mkopoguard.api import openapi, security
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
NOT_FROM_APP = "Request refused: send it from the MkopoGuard app"
SESSION_COOKIE = "mkopoguard_session"
APP_HEADER, APP_HEADER_VALUE = "X-Requested-With", "fetch"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


class Token(BaseModel):
    access_token: str | None = None  # None when the token was put in the session cookie
    token_type: Literal["bearer", "cookie"] = "bearer"
    expires_in: int  # seconds


def from_app(request: Request) -> bool:
    """True if the request came from the web app's own JavaScript (js/api.js)."""
    return request.headers.get(APP_HEADER) == APP_HEADER_VALUE


def set_session_cookie(response: Response, token: str, settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.access_token_minutes * 60,
        path="/",
        httponly=True,  # JavaScript can't read it
        samesite="strict",  # never sent with a request started by another site
        secure=settings.environment == "production",  # HTTPS only on the live site
    )


def clear_session_cookie(response: Response, settings) -> None:
    """Tell the browser to forget the session cookie (same name, path and flags as when set)."""
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="strict",
        secure=settings.environment == "production",
    )


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
    """The logged-in, active user, or 401. Put this on every route an officer uses.

    The Authorization header wins; without one, the session cookie is used.
    """
    if not token:
        token = request.cookies.get(SESSION_COOKIE)
        if token and request.method not in SAFE_METHODS and not from_app(request):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=NOT_FROM_APP)
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


@router.post("/login", response_model=Token, summary="Log in", responses=openapi.LOGIN_ERRORS)
def login(
    request: Request,
    response: Response,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    """Email (in the `username` field) and password in; a short-lived access token out.

    From the web app (header `X-Requested-With: fetch`) the token is set as an HttpOnly
    session cookie instead and the body has `access_token: null, token_type: "cookie"`.
    """
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
    expires_in = settings.access_token_minutes * 60
    if from_app(request):
        set_session_cookie(response, token, settings)
        return Token(token_type="cookie", expires_in=expires_in)
    return Token(access_token=token, expires_in=expires_in)


class SessionOut(BaseModel):
    user: UserOut | None = Field(description="The signed-in user, or null")


@router.get("/session", response_model=SessionOut, summary="Am I signed in")
def current_session(request: Request, session: SessionDep) -> SessionOut:
    """Who is signed in, or `{"user": null}`. Never 401, so the login page can check quietly.

    (The login page used to ask /auth/me, whose 401 for a signed-out visitor is a correct
    answer that browsers still report as a console error.)
    """
    scheme, _, header_token = request.headers.get("Authorization", "").partition(" ")
    token = header_token if scheme.lower() == "bearer" and header_token else None
    try:
        user = get_current_user(request, token, session)
    except HTTPException:
        return SessionOut(user=None)
    return SessionOut(user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut, summary="Who am I", responses=openapi.UNAUTHORIZED)
def me(user: CurrentUser) -> User:
    """The logged-in user's own details."""
    return user


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Log out",
)
def logout(request: Request) -> Response:
    """Clear the web app's session cookie. Works whether or not the session is still valid.

    Login tokens are not stored on the server, so a copy of a token (a bearer token from
    /docs, say) keeps working until it expires, at most `access_token_minutes` later.
    """
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(response, request.app.state.settings)
    log.info("Logout")
    return response
