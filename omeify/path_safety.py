"""File-identity checks and atomic installation for small text artifacts."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def paths_alias(left: str | Path, right: str | Path) -> bool:
    """Compare pathnames, symlink targets, and existing hard-link identities."""
    left, right = Path(left), Path(right)
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def check_output_path(
    path: str | Path, *, protected: Iterable[str | Path] = (), overwrite: bool = True,
) -> None:
    """Reject input aliases and unusable destinations without creating anything."""
    path = Path(path)
    for source in protected:
        if paths_alias(path, source):
            raise ValueError(f"Output {path} would replace a protected input/output: {source}")
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(f"Output already exists: {path}")
    if path.is_dir():
        raise IsADirectoryError(path)
    # Fail before long image work when an ancestor is a file or dangling link.
    for parent in path.parents:
        if os.path.lexists(parent):
            if not parent.is_dir():
                raise NotADirectoryError(parent)
            break


def atomic_write_text(
    path: str | Path, text: str, *, overwrite: bool = True,
    protected: Iterable[str | Path] = (),
) -> None:
    """Install complete UTF-8 text without truncating a destination's inode.

    A final-component symlink is replaced, never followed for writing. Parent
    directory symlinks are allowed. No-overwrite uses an atomic hard link on the
    destination filesystem and fails if hard links are unsupported. This is one
    artifact's transaction, not a transaction spanning an image and its report.
    """
    path = Path(path)
    protected = tuple(protected)  # Reused across preflight and installation.
    check_output_path(path, protected=protected, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".omeify-report-", suffix=".partial", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip("\n") + "\n")
        check_output_path(path, protected=protected, overwrite=overwrite)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
