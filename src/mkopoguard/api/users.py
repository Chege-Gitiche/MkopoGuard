"""Step 5.3: creating and managing accounts (used by scripts/create_user.py and the tests).

There is no public sign-up: a bank creates its officers' accounts itself.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from mkopoguard.api.models import ROLES, User
from mkopoguard.api.security import check_password_policy, hash_password


class UserExists(ValueError):
    pass


class UserNotFound(LookupError):
    pass


def find_user(session: Session, email: str) -> User | None:
    return session.scalars(select(User).where(User.email == email.strip().lower())).first()


def create_user(
    session: Session, email: str, full_name: str, password: str, role: str = "officer"
) -> User:
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}")
    if not full_name.strip():
        raise ValueError("Full name is required")
    if find_user(session, email):
        raise UserExists(f"An account for {email.strip().lower()} already exists")
    check_password_policy(password, email)
    user = User(
        email=email,
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    session.add(user)
    session.commit()
    return user


def set_active(session: Session, email: str, active: bool) -> User:
    """Deactivate (or reactivate) an account. Accounts with history are never deleted."""
    user = find_user(session, email)
    if user is None:
        raise UserNotFound(f"No account for {email}")
    user.is_active = active
    session.commit()
    return user


def set_password(session: Session, email: str, password: str) -> User:
    user = find_user(session, email)
    if user is None:
        raise UserNotFound(f"No account for {email}")
    check_password_policy(password, email)
    user.hashed_password = hash_password(password)
    session.commit()
    return user
