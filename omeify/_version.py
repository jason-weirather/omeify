from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_DISTRIBUTION_NAME = "omeify"


def _source_tree_version() -> str | None:
    """Read the authoritative version from a neighboring pyproject.toml.

    A regular wheel installation does not contain the repository-level
    pyproject.toml, so installed packages fall through to distribution
    metadata. Editable installs and direct source-tree imports use the same
    version value that the build backend reads.
    """

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.is_file():
        return None
    try:
        with pyproject_path.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        value = project["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _distribution_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


__version__ = _source_tree_version() or _distribution_version(_DISTRIBUTION_NAME) or "0+unknown"


def get_version_info() -> dict[str, Any]:
    """Return versions of omeify and its image I/O stack."""

    return {
        "omeify": __version__,
        "python": platform.python_version(),
        "tifffile": _distribution_version("tifffile"),
        "imagecodecs": _distribution_version("imagecodecs"),
        "numpy": _distribution_version("numpy"),
        "click": _distribution_version("click"),
        "lxml": _distribution_version("lxml"),
        "jsonschema": _distribution_version("jsonschema"),
        "ome-schema": _distribution_version("ome-schema"),
    }
