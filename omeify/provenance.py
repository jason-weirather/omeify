from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: str | Path) -> dict[str, str]:
    """Return MD5 and SHA-256 checksums for one file."""

    file_path = Path(path)
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5_checksum": md5.hexdigest(), "sha256_checksum": sha256.hexdigest()}


def readable_runtime(seconds: float) -> str:
    """Render elapsed seconds as ``HH:MM:SS.ss``."""

    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
