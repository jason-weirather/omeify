"""Actual Sheetbend registry/capability tests, with inference only replaced.

Core Sheetbend is optional. These tests do not require the LLM extra or network.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from omeify import intelligence as ai

sheetbend = pytest.importorskip("sheetbend", reason="Optional Sheetbend is not installed")


def registry_with(*scopes, default=None):
    definitions = {}
    for scope in scopes:
        value = {
            "protocol": "openai-compatible", "base_url": "http://127.0.0.1:1/v1",
            "scope": scope, "auth": {"type": "none"}, "default_model": "example",
            "models": {"example": {"capabilities": {"json_schema": True}}},
        }
        if scope == "institutional":
            value["organization"] = "example-lab"
        definitions[scope] = value
    data = {"schema_version": 2, "sources": definitions}
    if default:
        data["default_source"] = default
    return sheetbend.Registry.from_dict(data)


@pytest.fixture
def packet():
    records = [{
        "id": "m1", "locations": ["IFD[0]/DateTime"], "value": "2025:04:04 12:00:00",
        "truncated": False, "occurrences": 1,
    }]
    return {
        "records": records,
        "coverage": {
            "ifds_scanned": 1, "tags_scanned": 1, "records_available": 1,
            "records_included": 1, "records_truncated": 0,
            "metadata_chars": len(json.dumps(records, ensure_ascii=False, separators=(",", ":"))),
            "max_metadata_chars": 16000, "scan_limited": False,
            "omitted": {
                "binary_or_large_arrays": 0, "oversized_tags": 0,
                "unreadable_tags": 0, "embedded_binary_or_unsafe_xml": 0,
            }, "warnings": [],
        },
    }


@pytest.fixture
def connections(monkeypatch, packet):
    seen = []
    summary = {
        "overview": {"text": "A TIFF timestamp is present.", "record_ids": ["m1"]},
        "findings": [], "cautions": [],
    }

    @contextmanager
    def connect(self, model, **kwargs):
        seen.append((self.name, model, kwargs))
        response = SimpleNamespace(text=lambda: json.dumps(summary))
        yield SimpleNamespace(prompt=lambda *args, **kw: response)

    monkeypatch.setattr(sheetbend.source.Source, "_connect", connect)
    return seen


def test_institutional_precedes_local_even_with_local_default(packet, connections):
    result = ai.summarize_metadata(
        packet, registry=registry_with("local", "institutional", default="local"),
    )
    assert result["source"]["scope"] == "institutional"
    assert result["source"]["organization"] == "example-lab"
    assert connections[0][0] == "institutional"
    assert connections[0][2]["application"] == "omeify"
    assert connections[0][2]["tool"] == "inspect"
    assert connections[0][2]["reasoning"] is None  # Honor Sheetbend model configuration.


def test_local_is_selected_when_no_institutional_source_exists(packet, connections):
    result = ai.summarize_metadata(packet, registry=registry_with("local"))
    assert result["source"]["scope"] == "local"
    assert len(connections) == 1


def test_local_only_is_enforced(packet, connections):
    registry = registry_with("institutional", "local")
    result = ai.summarize_metadata(packet, registry=registry, allowed_scopes=["local"])
    assert result["source"]["scope"] == "local"
    with pytest.raises(ai.IntelligenceError, match="outside allowed_scopes"):
        ai.summarize_metadata(
            packet, registry=registry, source_name="institutional", allowed_scopes=["local"],
        )
    assert len(connections) == 1


@pytest.mark.parametrize("name", [None, "external"])
def test_external_is_never_implicitly_allowed(packet, connections, name):
    with pytest.raises(ai.IntelligenceError, match="allowed_scopes"):
        ai.summarize_metadata(packet, registry=registry_with("external"), source_name=name)
    assert not connections


def test_explicit_external_scope_is_deliberate(packet, connections):
    result = ai.summarize_metadata(
        packet, registry=registry_with("external"), allowed_scopes=["external"],
    )
    assert result["source"]["scope"] == "external"
    assert result["allowed_scopes"] == ["external"]


@pytest.mark.parametrize("capability", ["json_schema", "system_prompt"])
def test_capabilities_checked_before_credentials_and_connection(packet, connections, capability):
    data = registry_with("local").to_dict()
    data["sources"]["local"]["auth"] = {"type": "bearer", "env": "NEVER_LOOK_THIS_UP"}
    data["sources"]["local"]["models"]["example"]["capabilities"][capability] = False
    with pytest.raises(ai.IntelligenceError, match=capability):
        ai.summarize_metadata(packet, registry=sheetbend.Registry.from_dict(data))
    assert not connections


def test_empty_scopes_and_missing_registry_fail_closed(packet, connections):
    with pytest.raises(ai.IntelligenceError, match="allowed_scopes"):
        ai.summarize_metadata(packet, registry=registry_with("local"), allowed_scopes=[])
    with pytest.raises(ai.IntelligenceError, match="No configured source"):
        ai.summarize_metadata(packet, registry=registry_with())
    assert not connections


def test_explicit_config_override_uses_sheetbend_loader(packet, connections, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('''schema_version = 2
[sources.local]
protocol = "openai-compatible"
base_url = "http://127.0.0.1:1/v1"
scope = "local"
auth = {type = "none"}
default_model = "example"
[sources.local.models.example]
capabilities = {json_schema = true}
''')
    monkeypatch.setenv("SHEETBEND_CONFIG", str(config))
    result = ai.summarize_metadata(packet)
    assert result["source"]["name"] == "local"
    assert str(config) not in json.dumps(result)
    assert len(connections) == 1
