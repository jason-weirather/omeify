"""Actual optional Sheetbend/LLM/SDK integration against synthetic loopback HTTP.

The server returns controlled responses, not model inference. Run with the
intelligence extra installed; a skipped module is not an integration pass.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pytest
import tifffile

from omeify import TiffInspector
from omeify.intelligence import (
    IntelligenceError,
    metadata_question_schema,
    metadata_summary_schema,
)

sheetbend = pytest.importorskip("sheetbend", reason="Optional Sheetbend is not installed")
pytest.importorskip("llm", minversion="0.35", reason="Optional LLM integration is not installed")
pytest.importorskip("openai", minversion="3", reason="Optional SDK integration is not installed")


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEETBEND_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    state = {"requests": [], "mode": "valid"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": body})
            payload = json.loads(body["messages"][-1]["content"])
            question = payload.get("question")
            packet = payload if question is None else payload["metadata"]
            record = next(r for r in packet["records"] if r["value"] == "SYNTHETIC-SLIDE")
            summary = {
                "overview": {"text": "The metadata includes a slide label.",
                             "record_ids": [record["id"]]},
                "findings": [{
                    "record_id": record["id"], "category": "identifier", "label": "Slide label",
                    "interpretation": "This is a synthetic test label.",
                }],
                "cautions": [],
            }
            if question is not None:
                summary = {
                    "status": "answered",
                    "paragraphs": [{
                        "text": "The slide identifier is listed below.", "record_ids": [],
                    }],
                    "items": [{"label": "Slide identifier", "record_id": record["id"]}],
                    "cautions": [],
                }
            if state["mode"] == "bad-evidence":
                if question is None:
                    summary["overview"]["record_ids"][0] = "m999999"
                else:
                    summary["items"][0]["record_id"] = "m999999"
            content = "not JSON" if state["mode"] == "bad-json" else json.dumps(summary)
            response = {
                "id": "fixture-completion", "object": "chat.completion", "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            }
            status = 200
            if state["mode"] == "http-error":
                status = 429
                response = {"error": {"message": "DO_NOT_ECHO_PROVIDER_BODY", "type": "rate_limit"}}
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("question", [None, "What is the slide identifier?"])
@pytest.mark.parametrize("mode", ["valid", "bad-json", "bad-evidence", "http-error"])
def test_actual_request_schema_identity_lifetime_and_failure_modes(
    endpoint, tmp_path, mode, question,
):
    endpoint["mode"] = mode
    path = tmp_path / "DO_NOT_SEND_CURRENT_FILENAME.tif"
    tifffile.imwrite(
        path, np.zeros((8, 8), dtype=np.uint8), metadata=None,
        description="<Vendor><SlideID>SYNTHETIC-SLIDE</SlideID></Vendor>",
    )
    registry = sheetbend.Registry.from_dict({
        "schema_version": 2,
        "sources": {"fixture": {
            "protocol": "openai-compatible", "base_url": endpoint["url"],
            "scope": "local", "auth": {"type": "none"}, "default_model": "fixture-model",
            "models": {"fixture-model": {"capabilities": {"json_schema": True}}},
        }},
    })
    inspector = TiffInspector(path)
    if mode == "valid":
        result = inspector.summarize_metadata(registry=registry, question=question)
        if question is None:
            assert result["summary"]["findings"][0]["value"] == "SYNTHETIC-SLIDE"
        else:
            assert result["answer"]["items"][0]["value"] == "SYNTHETIC-SLIDE"
            assert result["question"] == question and "summary" not in result
        assert inspector.validation_errors() == ()
        inspector.to_json()
        inspector.render_text()
    else:
        with pytest.raises(IntelligenceError) as error:
            inspector.summarize_metadata(registry=registry, question=question)
        assert "DO_NOT_ECHO_PROVIDER_BODY" not in str(error.value)
        assert "intelligence" not in inspector.report
    assert len(endpoint["requests"]) == 1  # No catalog probes, retries, or render-time calls.
    request = endpoint["requests"][0]
    assert request["path"] == "/v1/chat/completions"
    wire = request["body"]
    assert wire["response_format"]["type"] == "json_schema"
    sent_payload = json.loads(wire["messages"][-1]["content"])
    sent_packet = sent_payload if question is None else sent_payload["metadata"]
    schema = metadata_summary_schema if question is None else metadata_question_schema
    assert wire["response_format"]["json_schema"]["schema"] == schema(
        record_ids=[record["id"] for record in sent_packet["records"]],
    )
    assert "tools" not in wire
    assert "DO_NOT_SEND_CURRENT_FILENAME" not in json.dumps(wire)
    runtime = sheetbend.Runtime().snapshot()
    assert not any(
        row["state"] in {"waiting", "running", "streaming"} for row in runtime["requests"]
    )
    assert any(
        row["application"] == "omeify" and row["tool"] == "inspect" for row in runtime["requests"]
    )
    assert "SYNTHETIC-SLIDE" not in json.dumps(runtime)
