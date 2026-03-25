from __future__ import annotations

"""
batch_run.py

Run the pipeline for a user-provided list of companies/domains, optionally repeated N times.
This makes it easy to generate many runs for diffs without hardcoding companies.

Examples
1) Provide companies inline (repeat twice):
python3 batch_run.py --companies "Goldman Sachs|goldmansachs.com,JPMorgan Chase|jpmorganchase.com" --repeat 2

2) Provide a file (one company per line):
python3 batch_run.py --file companies.txt --repeat 2

companies.txt format (one per line):
Goldman Sachs|goldmansachs.com
JPMorgan Chase|jpmorganchase.com
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class Company:
    name: str
    domain: str


def parse_companies_arg(s: str) -> List[Company]:
    # Format: "Name|domain,Name2|domain2"
    out: List[Company] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "|" not in p:
            raise ValueError(f"Bad company entry '{p}'. Expected Name|domain.")
        name, domain = [x.strip() for x in p.split("|", 1)]
        if not name or not domain:
            raise ValueError(f"Bad company entry '{p}'. Name and domain required.")
        out.append(Company(name=name, domain=domain))
    return out


def parse_companies_file(path: str) -> List[Company]:
    # Each non-empty non-comment line: Name|domain
    out: List[Company] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"Bad line '{line}'. Expected Name|domain.")
        name, domain = [x.strip() for x in line.split("|", 1)]
        out.append(Company(name=name, domain=domain))
    return out


def run_one(company: Company, run_id: str) -> None:
    cmd = [
        "python3",
        "run.py",
        "--company-name",
        company.name,
        "--company-domain",
        company.domain,
        "--run-id",
        run_id,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vitelis for multiple companies and repeated runs.")
    parser.add_argument(
        "--companies",
        default="",
        help='Inline list: "Name|domain,Name2|domain2"',
    )
    parser.add_argument(
        "--file",
        default="",
        help="Path to companies file. Each line: Name|domain",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="How many runs per company (default 2 so diffs work).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between runs (helps rate limits).",
    )
    args = parser.parse_args()

    if not args.companies and not args.file:
        raise SystemExit('Provide --companies or --file. Example: --companies "Goldman Sachs|goldmansachs.com"')

    companies: List[Company] = []
    if args.companies:
        companies.extend(parse_companies_arg(args.companies))
    if args.file:
        companies.extend(parse_companies_file(args.file))

    # Deduplicate exact duplicates while preserving order
    seen: set[Tuple[str, str]] = set()
    unique: List[Company] = []
    for c in companies:
        key = (c.name.lower(), c.domain.lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)

    for c in unique:
        for i in range(1, args.repeat + 1):
            # Deterministic run_id format so diffs are easy: shortdomain_r1, shortdomain_r2, etc.
            short = c.domain.split(".")[0]
            run_id = f"{short}_r{i}"
            run_one(c, run_id)
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()