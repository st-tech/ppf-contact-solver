#!/usr/bin/env python3
"""Fetch cbor2 wheels listed in blender_manifest.toml from PyPI.

The Blender extension manifest references six cbor2 wheels (cp311/cp313
times macOS arm64 / manylinux x86_64 / Windows amd64). They are NOT
checked into the repo; CI and local installs run this script to pull
them from files.pythonhosted.org and verify each against the SHA-256
digest published on PyPI.

Idempotent: skips files already present whose hash matches.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

CBOR2_VERSION = "6.0.1"

# (filename, packages-prefix on files.pythonhosted.org, sha256)
# Hashes taken from https://pypi.org/pypi/cbor2/6.0.1/json on 2026-05-07.
WHEELS: list[tuple[str, str, str]] = [
    (
        "cbor2-6.0.1-cp311-cp311-macosx_11_0_arm64.whl",
        "32/7d/b2f9cd0c27bce0415a7dec71d0073e29b8e3f5bf45ea25f6874392c24add",
        "4d8dba16aa67ca13aa85849c5cbe4a88a353d6ed28ca8c11afc2ad9bc96b7ea7",
    ),
    (
        "cbor2-6.0.1-cp311-cp311-manylinux_2_28_x86_64.whl",
        "1f/66/fbbdf6924848f2c31632e6801c0b109216bacbc278ebb11c21ad4b312336",
        "873ac665a1e8b3b9baa7d9384221917c828e8c72f670b2df887ed4a627367842",
    ),
    (
        "cbor2-6.0.1-cp311-cp311-win_amd64.whl",
        "a8/e5/004946ebd82db48fc72cb18a0eea2f720de97dd3f5c7e87a5f2f716f1ed4",
        "ce23169d812f37636dbf92af67460a4eee5c340c4b838b883e307ac1cde9f67e",
    ),
    (
        "cbor2-6.0.1-cp313-cp313-macosx_11_0_arm64.whl",
        "08/ee/d11300317773bc8e85e23f59fc71c732ba1176d059341588318cab81f501",
        "067d23ac75bfa35bed0e795169139259dc9d9bae503c8ede29740f99b37415f3",
    ),
    (
        "cbor2-6.0.1-cp313-cp313-manylinux_2_28_x86_64.whl",
        "8f/d0/3ffca18a38f0d8b6b1a6649d1245b876f5cf4cf9be3f5b2d19717b227af8",
        "50ebae27b72061c8baf3cd8458c3eb2de7c112d0be77af24e8c4206a2b0e7b61",
    ),
    (
        "cbor2-6.0.1-cp313-cp313-win_amd64.whl",
        "9f/3e/c86f51bc78c211bcf685485a8c888713d714ebd64192435a45b68bef2b0b",
        "897f6fe58d1522608b6b71a7aa964f31c40deed5fff2d00511233bacb396dded",
    ),
]

DEST = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(filename: str, prefix: str, sha: str) -> None:
    out = DEST / filename
    if out.exists() and sha256_file(out) == sha:
        print(f"ok (cached)  {filename}")
        return
    url = f"https://files.pythonhosted.org/packages/{prefix}/{filename}"
    print(f"download     {filename}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha:
        sys.exit(
            f"hash mismatch for {filename}\n"
            f"  expected {sha}\n"
            f"  got      {actual}"
        )
    out.write_bytes(data)
    print(f"ok           {filename}  ({len(data) // 1024} KiB)")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for spec in WHEELS:
        fetch_one(*spec)
    print(f"\n{len(WHEELS)} wheels in {DEST}")


if __name__ == "__main__":
    main()
