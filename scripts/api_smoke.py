"""Step 5.8: run every endpoint once against a running server, and check each answer.

The automated twin of docs/api_checklist.md. Use it after any change, and in Phase 8 against
the live URL:

  python scripts/api_smoke.py --email emmanuel@mkopoguard.app
  python scripts/api_smoke.py --base-url https://<your-app>.onrender.com --email demo@...

The password is read at a hidden prompt (or from MKOPOGUARD_SMOKE_PASSWORD). It makes exactly
one wrong login on purpose, far below the rate limit, and scores 2 samples. For an officer
account those are saved to its history; the demo account saves nothing.
"""

import argparse
import getpass
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
PASSWORD_ENV = "MKOPOGUARD_SMOKE_PASSWORD"
EXPECTED_PATHS = {
    "/health",
    "/auth/login",
    "/auth/me",
    "/predict",
    "/model/info",
    "/predictions",
    "/predictions/{prediction_id}",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def run(client: httpx.Client, email: str, password: str) -> list[Check]:
    checks: list[Check] = []

    def check(name: str, condition: bool, detail: str = "") -> bool:
        checks.append(Check(name, bool(condition), detail))
        return bool(condition)

    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    thin = next(s for s in manifest if s["statement"] is None)
    holder = next(s for s in manifest if s["statement"])

    # --- public ---
    r = client.get("/health")
    check("GET /health is up", r.status_code == 200 and r.json().get("status") == "ok", r.text[:80])
    check("Every response has a request id", len(r.headers.get("X-Request-ID", "")) == 16)
    check("Security headers are set", r.headers.get("X-Content-Type-Options") == "nosniff")
    r = client.get("/openapi.json")
    paths = set(r.json().get("paths", {})) if r.status_code == 200 else set()
    check("API docs list every endpoint", EXPECTED_PATHS <= paths, str(EXPECTED_PATHS - paths))

    # --- auth ---
    r = client.post("/auth/login", data={"username": email, "password": password + "-wrong"})
    check(
        "Wrong password: 401 with the generic message",
        r.status_code == 401 and r.json().get("detail") == "Incorrect email or password",
    )
    check("No login: 401", client.get("/auth/me").status_code == 401)
    r = client.post("/auth/login", data={"username": email, "password": password})
    if not check("Login works", r.status_code == 200, r.text[:120]):
        return checks  # nothing else can run without a token
    auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = client.get("/auth/me", headers=auth)
    me = r.json() if r.status_code == 200 else {}
    check(
        "GET /auth/me shows you, without the password hash",
        me.get("email") == email.strip().lower() and "hashed_password" not in me,
    )
    check("Personal responses are not cached", r.headers.get("Cache-Control") == "no-store")
    r = client.get("/model/info", headers=auth)
    check("GET /model/info", r.status_code == 200 and "version" in r.json(), r.text[:80])

    # --- scoring ---
    def score(sample: dict, profile_text: str | None = None, statement: bytes | None = None):
        profile_text = profile_text or (SAMPLES / sample["profile"]).read_text(encoding="utf-8")
        if statement is None and sample["statement"]:
            statement = (SAMPLES / sample["statement"]).read_bytes()
        files = {"statement": ("statement.csv", statement, "text/csv")} if statement else None
        return client.post("/predict", headers=auth, data={"profile": profile_text}, files=files)

    made = []
    for sample in (thin, holder):
        r = score(sample)
        body = r.json() if r.status_code == 200 else {}
        ok = (
            r.status_code == 200
            and abs(body.get("probability_of_default", -1) - sample["probability_of_default"])
            < 1e-9
        )
        check(
            f"POST /predict {sample['name']}: {sample['probability_of_default']:.1%}, "
            f"{sample['decision']}",
            ok and body.get("decision") == sample["decision"],
            r.text[:120],
        )
        check(
            "  ...with 3 + 2 reasons",
            len(body.get("reasons", {}).get("raising", [])) == 3
            and len(body.get("reasons", {}).get("lowering", [])) == 2,
        )
        made.append(body)

    profile = json.loads((SAMPLES / thin["profile"]).read_text(encoding="utf-8")) | {"age": 15}
    r = score(thin, json.dumps(profile))
    check(
        "Bad profile: 422 naming the field",
        r.status_code == 422 and r.json()["detail"][0]["loc"] == ["profile", "age"],
        r.text[:120],
    )
    lines = (SAMPLES / holder["statement"]).read_text(encoding="utf-8").splitlines()
    lines[5] = lines[5].replace(",out,", ",sideways,").replace(",in,", ",sideways,")
    r = score(holder, statement=("\n".join(lines) + "\n").encode())
    check(
        "Bad statement: 422 naming the row",
        r.status_code == 422 and r.json()["detail"][0]["loc"] == ["statement", "row 6"],
        r.text[:120],
    )

    # --- history ---
    r = client.get("/predictions", headers=auth, params={"size": 5})
    check("GET /predictions", r.status_code == 200 and "total" in r.json(), r.text[:80])
    ids = [i["id"] for i in r.json().get("items", [])] if r.status_code == 200 else []
    if me.get("role") == "demo":
        check("Demo account: nothing was saved", all(m.get("id") is None for m in made))
    else:
        newest = made[-1].get("id")
        check("The new prediction is first in your history", ids[:1] == [newest], str(ids[:3]))
        r = client.get(f"/predictions/{newest}", headers=auth)
        check(
            "GET /predictions/{id}",
            r.status_code == 200
            and r.json().get("probability_of_default") == made[-1].get("probability_of_default"),
        )
    r = client.get("/predictions/999999999", headers=auth)
    check(
        "Someone else's or a missing prediction: 404",
        r.status_code == 404 and r.json() == {"detail": "Prediction not found"},
    )
    return checks


def report(checks: list[Check]) -> int:
    for c in checks:
        print(
            f"{'PASS' if c.passed else 'FAIL'}  {c.name}"
            + (f"   ({c.detail})" if not c.passed and c.detail else "")
        )
    failed = sum(not c.passed for c in checks)
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running MkopoGuard API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    password = os.environ.get(PASSWORD_ENV) or getpass.getpass("Password (hidden): ")
    with httpx.Client(base_url=args.base_url, timeout=60) as client:  # free hosts wake slowly
        return report(run(client, args.email, password))


if __name__ == "__main__":
    sys.exit(main())
