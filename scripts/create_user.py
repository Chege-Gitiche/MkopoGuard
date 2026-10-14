"""Step 5.3: create, deactivate or reactivate accounts. There is no public sign-up.

Run from the project root (in Git Bash, put `winpty` in front so the password prompt works):

  winpty python scripts/create_user.py create --email asha@bank.co.ke --name "Asha Wanjiru"
  winpty python scripts/create_user.py create --email demo@mkopoguard.app --name "Demo" --role demo
  python scripts/create_user.py deactivate --email asha@bank.co.ke
  python scripts/create_user.py reactivate --email asha@bank.co.ke
  winpty python scripts/create_user.py password --email asha@bank.co.ke

The password is typed at a hidden prompt (twice), never on the command line, where it would be
saved in the shell history. For hosts with no terminal (Phase 8), set MKOPOGUARD_NEW_PASSWORD
for that one command instead. Uses the same database as the API (MKOPOGUARD_DATABASE_URL / .env).
"""

import argparse
import getpass
import os
import sys

from mkopoguard.api import db
from mkopoguard.api.models import ROLES
from mkopoguard.api.security import WeakPassword, check_password_policy
from mkopoguard.api.settings import DatabaseSettings
from mkopoguard.api.users import UserExists, UserNotFound, create_user, set_active, set_password

PASSWORD_ENV = "MKOPOGUARD_NEW_PASSWORD"


def read_password(email: str) -> str:
    if os.environ.get(PASSWORD_ENV):
        password = os.environ[PASSWORD_ENV]
    else:
        password = getpass.getpass("Password (hidden): ")
        if getpass.getpass("Same password again: ") != password:
            raise SystemExit("The two passwords don't match; nothing was changed")
    try:
        check_password_policy(password, email)
    except WeakPassword as error:
        raise SystemExit(f"{error}; nothing was changed") from None
    return password


def parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage MkopoGuard accounts.")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create", help="create an account")
    create.add_argument("--email", required=True)
    create.add_argument("--name", required=True, help="full name")
    create.add_argument("--role", choices=ROLES, default="officer")
    for action in ("deactivate", "reactivate", "password"):
        sub.add_parser(action, help=f"{action} an account").add_argument("--email", required=True)
    return parser.parse_args(argv)


def main(argv=None, url: str | None = None) -> int:
    args = parse(argv)
    engine = db.make_engine(url or DatabaseSettings().database_url)
    db.check_schema(engine)  # refuse to write to a database that isn't migrated
    with db.make_sessions(engine)() as session:
        try:
            if args.action == "create":
                user = create_user(
                    session, args.email, args.name, read_password(args.email), args.role
                )
                print(f"Created {user.role} account {user.email} (id {user.id})")
            elif args.action == "password":
                user = set_password(session, args.email, read_password(args.email))
                print(f"Password changed for {user.email}")
            else:
                user = set_active(session, args.email, args.action == "reactivate")
                print(f"{user.email} is now {'active' if user.is_active else 'deactivated'}")
        except (UserExists, UserNotFound, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
