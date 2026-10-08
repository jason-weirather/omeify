"""Validation of current, packaged operation-report contracts without network I/O."""
from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource


@lru_cache(maxsize=1)
def _resources() -> dict[str, dict[str, Any]]:
    # Shared documents and compiled validators are reused across series/crops.
    return {
        item.name: json.loads(item.read_text(encoding="utf-8"))
        for item in files("omeify.schemas").iterdir()
        if item.name.endswith(".schema.json")
    }


def load_schema(name: str) -> dict[str, Any]:
    """Return an independently editable copy of one packaged schema document."""
    return deepcopy(_resources()[name])


@lru_cache(maxsize=1)
def _registry() -> Registry:
    # An explicit registry never falls back to fetching remote $id/$ref URLs.
    return Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema))
        for schema in _resources().values() if "$id" in schema
    )


@lru_cache(maxsize=None)
def report_validator(name: str) -> Draft202012Validator:
    """Resolve a current contract and all its references from installed resources."""
    schema = _resources()[name]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_registry())


def validate_report(report: dict[str, Any], schema_name: str) -> None:
    """Validate serialized structure, identifying errors without echoing report values."""
    try:
        report_validator(schema_name).validate(report)
    except ValidationError as exc:
        raise ValueError(
            f"{schema_name} at {exc.json_path}: failed {exc.validator} validation"
        ) from exc


def complete_report(report: dict[str, Any], schema_name: str) -> dict[str, Any]:
    """Add the schema-owned identifier where defined, and validate a new report."""
    schema = _resources()[schema_name]
    marker = schema.get("properties", {}).get("schema", {}).get("const")
    result = {"schema": marker, **report} if marker is not None else dict(report)
    validate_report(result, schema_name)
    return result
