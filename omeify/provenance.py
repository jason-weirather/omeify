from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .progress import ProgressLogger

LOGGER = logging.getLogger(__name__)


def hash_file(
    path: str | Path,
    *,
    progress_label: str | None = None,
) -> dict[str, str]:
    """Return MD5 and SHA-256 checksums for one file in a single read pass."""

    file_path = Path(path)
    progress = (
        None
        if progress_label is None
        else ProgressLogger(
            LOGGER,
            progress_label,
            file_path.stat().st_size,
            unit="bytes",
        )
    )
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    completed = 0
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            completed += len(chunk)
            if progress is not None:
                progress.update(completed)
    if progress is not None:
        progress.finish()
    return {"md5_checksum": md5.hexdigest(), "sha256_checksum": sha256.hexdigest()}


def readable_runtime(seconds: float) -> str:
    """Render elapsed seconds as ``HH:MM:SS.ss``."""

    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
