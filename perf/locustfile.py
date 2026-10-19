"""Step 7.4: load test of POST /predict (NFR-16, test case TC-70).

Each simulated officer signs in once, then keeps scoring the ten sample applicants (eight with
a statement, two without), pausing 1 to 2 seconds between them like someone reading a result.
Run it through perf/run_load_test.py, which starts a throwaway copy of the app, passes the
login in environment variables (never on the command line), and checks the result.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MANIFEST = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
APPLICANTS = [
    (
        (SAMPLES / entry["profile"]).read_text(encoding="utf-8"),
        (SAMPLES / entry["statement"]).read_bytes() if entry["statement"] else None,
    )
    for entry in MANIFEST
]


class Officer(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        response = self.client.post(
            "/auth/login",
            data={
                "username": os.environ["MKOPOGUARD_LOAD_EMAIL"],
                "password": os.environ["MKOPOGUARD_LOAD_PASSWORD"],
            },
            name="/auth/login",
        )
        response.raise_for_status()
        self.client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    @task(5)
    def assess(self):
        profile, statement = random.choice(APPLICANTS)
        files = {"statement": ("statement.csv", statement, "text/csv")} if statement else None
        self.client.post("/predict", data={"profile": profile}, files=files, name="/predict")

    @task(1)
    def history(self):
        self.client.get("/predictions", name="/predictions")
