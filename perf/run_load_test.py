"""Step 7.4: run the /predict load test against a throwaway copy of the app and check it.

    python perf/run_load_test.py                 10 users for 2 minutes (the release check)
    python perf/run_load_test.py --minutes 0.5   a quick try

Starts the app on a temporary database (mkopoguard/local_server.py), runs Locust headless with
perf/locustfile.py, and writes the result to docs/results/load_test.json. Fails (exit code 1)
if any request failed or /predict's 95th percentile is at or over the target.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mkopoguard import config
from mkopoguard.local_server import local_server

LOCUSTFILE = Path(__file__).resolve().parent / "locustfile.py"
RESULT = config.PROJECT_ROOT / "docs" / "results" / "load_test.json"
P95_TARGET_MS = 500  # local; the live free tier's target (1.5 s) is checked in Phase 8


def read_stats(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["Name"]: row for row in csv.DictReader(f)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=2.0)
    args = parser.parse_args()

    with local_server() as server, tempfile.TemporaryDirectory() as folder:
        print(f"Load test: {args.users} users for {args.minutes:g} minutes on {server.url}")
        prefix = Path(folder) / "run"
        env = os.environ | {
            "MKOPOGUARD_LOAD_EMAIL": server.email,
            "MKOPOGUARD_LOAD_PASSWORD": server.password,
        }
        command = [
            sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
            "--host", server.url, "--users", str(args.users), "--spawn-rate", str(args.users),
            "--run-time", f"{round(args.minutes * 60)}s", "--csv", str(prefix),
            "--only-summary", "--loglevel", "WARNING",
        ]  # fmt: skip
        subprocess.run(command, env=env, check=False)
        stats = read_stats(Path(f"{prefix}_stats.csv"))

    predict, total = stats["/predict"], stats["Aggregated"]
    result = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "where": "local",
        "users": args.users,
        "seconds": round(args.minutes * 60),
        "model_version": config.MODEL_VERSION,
        "predict_requests": int(predict["Request Count"]),
        "predict_p50_ms": float(predict["50%"]),
        "predict_p95_ms": float(predict["95%"]),
        "predict_p99_ms": float(predict["99%"]),
        "predict_max_ms": round(float(predict["Max Response Time"])),
        "requests_per_second": round(float(total["Requests/s"]), 1),
        "failures": int(total["Failure Count"]),
        "p95_target_ms": P95_TARGET_MS,
        "machine": f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, "
        f"Python {platform.python_version()}",
    }
    passed = result["failures"] == 0 and result["predict_p95_ms"] < P95_TARGET_MS
    result["passed"] = passed
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"/predict: {result['predict_requests']} requests, p50 {result['predict_p50_ms']:.0f} ms, "
        f"p95 {result['predict_p95_ms']:.0f} ms (target under {P95_TARGET_MS}), "
        f"p99 {result['predict_p99_ms']:.0f} ms; {result['failures']} failures"
    )
    print(f"Saved {RESULT.relative_to(config.PROJECT_ROOT).as_posix()}")
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
