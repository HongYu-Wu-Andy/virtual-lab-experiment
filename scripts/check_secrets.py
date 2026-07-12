#!/usr/bin/env python3
"""Fail safely when provider credentials or private-key material enter the repository."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "provider API key": re.compile(rb"\bsk-(?:or-v1-|proj-|ant-)?[A-Za-z0-9_-]{16,}\b"),
    "GitHub token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "Google API key": re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "bearer token": re.compile(rb"(?i)\bBearer\s+([A-Za-z0-9._~-]{20,})"),
    "credential assignment": re.compile(
        rb'''(?ix)[\"']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)'''
        rb'''[\"']?\s*[=:]\s*[\"']([^\"'\r\n]{8,})[\"']'''
    ),
}
FIXTURES = {
    b"secret-value",
    b"must-not-survive",
    b"prompt-secret",
    b"your-api-key",
    b"replace-me",
    b"example-secret",
    b"<hidden>",
}


def tracked_files() -> Iterable[tuple[str, bytes]]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode(errors="replace")
        data = (ROOT / path).read_bytes()
        if b"\0" not in data[:8192]:
            yield path, data


def historical_blobs() -> Iterable[tuple[str, bytes]]:
    commits = subprocess.check_output(["git", "rev-list", "--all"], cwd=ROOT, text=True)
    seen: set[str] = set()
    for commit in commits.splitlines():
        tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit], cwd=ROOT)
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            blob = metadata.split()[2].decode()
            if blob in seen:
                continue
            seen.add(blob)
            data = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
            if b"\0" not in data[:8192]:
                path = raw_path.decode(errors="replace")
                yield f"{commit[:12]}:{path}", data


def is_fixture(value: bytes) -> bool:
    normalized = value.strip().lower()
    return normalized in FIXTURES or any(
        marker in normalized
        for marker in (b"example", b"placeholder", b"your-", b"your_", b"${")
    )


def scan(items: Iterable[tuple[str, bytes]]) -> list[str]:
    findings: list[str] = []
    for location, data in items:
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(data):
                value = match.group(1) if match.lastindex else match.group(0)
                if is_fixture(value):
                    continue
                line = data.count(b"\n", 0, match.start()) + 1
                fingerprint = hashlib.sha256(value).hexdigest()[:12]
                findings.append(
                    f"{name}: {location}:{line} length={len(value)} sha256={fingerprint}"
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also scan every unique Git blob")
    args = parser.parse_args()

    findings = scan(tracked_files())
    if args.history:
        findings.extend(scan(historical_blobs()))

    if findings:
        print("Potential credentials found; values are intentionally redacted:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Secret audit passed: no non-fixture credentials detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
