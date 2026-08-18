from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

__version__ = "0.4.0"


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def get_version_info() -> dict[str, Any]:
    """Return versions of omeify and its image I/O stack.

    Importing this module never shells out to external programs.  In particular,
    there is no Java tool discovery in the pure-Python implementation.
    """

    return {
        "omeify": _distribution_version("omeify") or __version__,
        "python": platform.python_version(),
        "tifffile": _distribution_version("tifffile"),
        "imagecodecs": _distribution_version("imagecodecs"),
        "click": _distribution_version("click"),
        "lxml": _distribution_version("lxml"),
        "ome-schema": _distribution_version("ome-schema"),
    }
