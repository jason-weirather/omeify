from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import cast

from .reports import validate_report

MAX_PROVENANCE_BYTES = 1024 * 1024


def _json_snapshot(value: object, ancestors: tuple[int, ...] = ()) -> object:
    """Copy data only; never stringify objects or coerce mapping keys."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("provenance must contain only finite JSON-serializable values")
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError("provenance must contain only finite JSON-serializable values")
    if id(value) in ancestors or len(ancestors) >= 64:
        raise ValueError("provenance must be acyclic and at most 64 containers deep")
    ancestors = (*ancestors, id(value))
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("provenance mapping keys must be plain strings")
        return {key: _json_snapshot(child, ancestors) for key, child in value.items()}
    return [_json_snapshot(child, ancestors) for child in value]


def canonical_provenance(
    provenance: Mapping[str, object] | None,
) -> tuple[str | None, dict[str, object] | None]:
    """Serialize explicit collection context, not automatically inherited metadata.

    Names/free text in this caller-supplied object are retained without privacy
    classification. The bound and known crop schema do not certify sharing safety.
    """
    if provenance is None:
        return None, None
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    value = cast(dict[str, object], _json_snapshot(provenance))
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))
    if len(rendered.encode("utf-8")) > MAX_PROVENANCE_BYTES:
        raise ValueError("Collection provenance exceeds 1 MiB")
    if value.get("schema") == "omeify.crop_provenance/1":
        validate_report(value, "crop_provenance.schema.json")
    return rendered, value


def readable_runtime(seconds: float) -> str:
    """Render elapsed seconds as ``HH:MM:SS.ss``."""

    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
