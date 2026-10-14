"""Step 5.3: passwords and login tokens.

Passwords
- Hashed with bcrypt (a slow, salted hash: a stolen database doesn't reveal passwords, and
  guessing them is expensive). The `bcrypt` library is used directly: passlib, which the guide
  names, is no longer maintained and breaks with bcrypt 4.1+.
- bcrypt only reads the first 72 bytes, and bcrypt 5 refuses anything longer, so the policy
  caps passwords at 72 bytes instead of silently ignoring the rest.
- Policy (NIST SP 800-63B style): at least 12 characters, not a common password, not the
  email address. No forced symbols or digits: length matters more.

Tokens
- Short-lived JWTs (30 minutes by default), signed with the secret key (HS256). They carry
  only the user's id and role; the user is still looked up on every request, so deactivating
  an account takes effect at once, not when the token expires.
- Issuer and audience are checked, the algorithm is pinned (an "alg: none" token is refused),
  and expiry is required.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

ALGORITHM = "HS256"
ISSUER = "mkopoguard"
AUDIENCE = "mkopoguard-api"
BCRYPT_ROUNDS = 12  # about a quarter of a second per hash; tests lower it to stay fast
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_BYTES = 72
COMMON_PASSWORDS = {
    "password1234",
    "password12345",
    "password123456",
    "123456789012",
    "qwertyuiop12",
    "iloveyou1234",
    "mkopoguard123",
    "mkopoguard2026",
    "letmein12345",
    "welcome12345",
    "administrator",
}


class WeakPassword(ValueError):
    """The password doesn't meet the policy."""


class InvalidToken(Exception):
    """The token is missing, expired, tampered with or not ours."""


# --- passwords -------------------------------------------------------------------------------


def check_password_policy(password: str, email: str = "") -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPassword(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise WeakPassword(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
    if password.lower() in COMMON_PASSWORDS:
        raise WeakPassword("Password is too common")
    if email and password.lower() == email.strip().lower():
        raise WeakPassword("Password must not be the email address")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    """True only for the right password; False (never an error) for anything else."""
    data = password.encode("utf-8")
    if len(data) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(data, hashed.encode("ascii"))
    except ValueError:  # not a bcrypt hash
        return False


_DUMMY_HASH: str | None = None


def dummy_hash() -> str:
    """Checked against when the email is unknown, so a wrong email takes as long as a wrong
    password: response time doesn't reveal which emails have accounts."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("not-a-real-password-for-timing-only")
    return _DUMMY_HASH


# --- tokens ------------------------------------------------------------------------------------


def create_access_token(
    user_id: int, role: str, secret: str, minutes: int, now: datetime | None = None
) -> str:
    now = now or datetime.now(UTC)
    claims = {
        "sub": str(user_id),
        "role": role,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict:
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
    except jwt.PyJWTError as error:
        raise InvalidToken(str(error)) from error
