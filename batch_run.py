from __future__ import annotations

"""
batch_run.py

Purpose
- Run the pipeline for multiple companies so you have more than one dataset.
- Run the same company twice so Run Diffs can compare two runs.
- Keep runs more repeatable by setting env defaults (temperature, delay).

How to use
- Activate venv
- Ensure .env has OPENAI_API_KEY
- Run: python3 batch_run.py
"""

import os
import subprocess
import time
import uuid
from typing import List, Tuple

# List of (display name, domain) to run.
# Add or remove companies here.
COMPANIES: List[Tuple[str, str]] = [
    ("Goldman Sachs", "goldmansachs.com"),
    ("JPMorgan Chase", "jpmorganchase.com"),
    ("Morgan Stanley", "morganstanley.com"),
    ("BlackRock", "blackrock.com"),
    ("Bloomberg", "bloomberg.com"),
]


def run_one(company_name: str, company_domain: str, run_id: str, max_urls: int) -> None:
    """
    Runs run.py once using subprocess so this script can loop over many runs.
    """
    cmd = [
        "python3",
        "run.py",
        "--company-name",
        company_name,
        "--company-domain",
        company_domain,
        "--run-id",
        run_id,
        "--max-urls",
        str(max_urls),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    """
    Main runner.
    - Sets defaults for repeatability.
    - Runs each company once.
    - Runs one extra repeat run for Goldman Sachs so diffs are immediately demoable.
    """
    # Repeatability knobs.
    # These only affect the pipeline if the code reads these env vars.
    os.environ.setdefault("OPENAI_TEMPERATURE", "0")
    os.environ.setdefault("OPENAI_MIN_DELAY_SECONDS", "10")

    # URL cap to help avoid 429 rate limits. Override by setting MAX_URLS in your shell.
    max_urls = int(os.environ.get("MAX_URLS", "40"))

    # First pass for all companies.
    for name, domain in COMPANIES:
        short = domain.split(".")[0]
        run_id = f"{short}_{str(uuid.uuid4())[:6]}"
        run_one(name, domain, run_id, max_urls=max_urls)

        # Small pause between runs to reduce rate limit spikes.
        time.sleep(2)

    # Second pass for a single company, so you can diff two runs for the same domain.
    run_one(
        "Goldman Sachs",
        "goldmansachs.com",
        f"gs_{str(uuid.uuid4())[:6]}",
        max_urls=max_urls,
    )


if __name__ == "__main__":
    main()