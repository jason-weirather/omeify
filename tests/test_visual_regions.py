"""Offline geometry/display/request tests; live adapter coverage is separately optional."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import ImageLevel, ImageMetadata, ImageSource, MultichannelImage, PixelSize
from omeify.cli import main
from omeify.intelligence import IntelligenceError
from omeify.preview import build_preview
from omeify.regions import parse_regions
from omeify.visual_intelligence import _materialize, locate_regions


def runner():
    options = {"mix_stderr": False} if "mix_stderr" in signature(CliRunner).parameters else {}
    return CliRunner(**options)


def response(bbox=(600, 100, 900, 900), size=None):
    return json.dumps({"status": "located", "message": "Approximate test location.",
                       "regions": [{"name": "right tissue", "bbox": list(bbox), "size_px": size}]})


def test_normalized_coordinates_and_fixed_dimensions():
    features, status, _ = _materialize(response(), width=100_000, height=50_000, roi_size=None)
    assert status == "located"
    assert parse_regions({"type": "FeatureCollection", "features": features})[0].bounds == (
        60000, 5000, 90000, 45000)
    features, _, _ = _materialize(response((990, 990, 1000, 1000), [2048, 2048]),
                                  width=100_001, height=50_003, roi_size=None)
    region = parse_regions(features)[0]
    assert region.bounds == (100001 - 2048, 50003 - 2048, 100001, 50003)
    assert features[0]["properties"]["edge_shift_xy"] != [0, 0]
    features, _, _ = _materialize(response(size=[100, 100]), width=9999, height=7001,
                                  roi_size=(2048, 1024))
    x0, y0, x1, y1 = parse_regions(features)[0].bounds
    assert (x1 - x0, y1 - y0) == (2048, 1024)  # CLI/library override is authoritative.


@pytest.mark.parametrize("bad", [
    response((900, 0, 100, 500)), response((0, 0, 1001, 500)), response(size=[10001, 100]),
    response((False, 0, 100, 100)), response(size=[3.5, 5]),
    response().replace('"located"', '"unavailable"'),
    response().replace('"right tissue"', '"  "'),
    response().replace('600', '1e400'),
    '```json\n' + response() + '\n```',
    '{"status":"located","status":"unavailable","regions":[],"message":""}',
])
def test_invalid_model_geometry_is_not_repaired(bad):
    with pytest.raises((IntelligenceError, ValueError)):
        _materialize(bad, width=10000, height=7000, roi_size=None)


def test_unavailable_is_explicit_not_fabricated_region():
    features, status, message = _materialize(
        '{"status":"unavailable","regions":[],"message":"No identifiable tissue"}',
        width=1000, height=1000, roi_size=None,
    )
    assert features == [] and status == "unavailable" and message


def test_binned_mean_preview_matches_independent_bins(image_factory):
    a = np.arange(259 * 515, dtype=np.float64).reshape(259, 515)
    image = image_factory(a)
    pixels, context = build_preview(image, max_size=128, display_range=(0, a.max()))
    ph, pw = pixels.shape
    y = np.floor((np.arange(259) + .5) / (259 / ph)).astype(int)
    x = np.floor((np.arange(515) + .5) / (515 / pw)).astype(int)
    indices = (y[:, None] * pw + x[None, :]).ravel()
    means = np.bincount(indices, weights=a.ravel()) / np.bincount(indices)
    expected = np.rint(means.reshape(ph, pw) / a.max() * 255).astype(np.uint8)
    np.testing.assert_array_equal(pixels, expected)
    assert context["base_pixels_per_preview_pixel_xy"] == [515 / pw, 259 / ph]


def test_huge_source_uses_known_pyramid_without_base_reads():
    class Virtual(ImageSource):
        def __init__(self):
            self.calls = []
            h, w = 100001, 180003
            size = PixelSize(.5, .5, "µm")
            super().__init__(ImageMetadata(
                axes="CYX", shape=(2, h, w), dtype=np.dtype("uint16"),
                channel_names=("CD3", "DAPI"), pixel_size=size,
                levels=(ImageLevel(0, "CYX", (2, h, w), size),
                        ImageLevel(1, "CYX", (2, 782, 1407), size.scaled(128), (128, 128))),
            ))

        def _read_region(self, y0, y1, x0, x1, *, level, channels):
            assert level == 1 and channels == (1,)
            assert y1 - y0 <= 1024 and x1 - x0 <= 1024
            self.calls.append((y0, y1, x0, x1))
            return np.broadcast_to(np.arange(x0, x1, dtype=np.uint16),
                                   (1, y1 - y0, x1 - x0)).copy()

    source = Virtual()
    with MultichannelImage(source) as image:
        preview, context = build_preview(image, max_size=128)
    assert preview.shape == (71, 128) and preview.dtype == np.uint8
    assert context["source_level"] == 1 and context["source_downsample_yx"] == [128., 128.]
    assert context["base_width"] == 180003 and context["channel_selection"] == "unique_dapi"
    assert len(source.calls) == 2


def test_unknown_pyramid_scale_is_not_guessed_and_nan_is_reported():
    a = np.zeros((255, 383), dtype=np.float32)
    a[100:150, 200:300] = 10
    a[0, 0] = np.nan
    levels = (ImageLevel(0, "YX", a.shape), ImageLevel(1, "YX", a[::2, ::2].shape))
    with MultichannelImage.from_array(a, axes="YX", level_arrays=(a[::2, ::2],),
                                     levels=levels) as image:
        _, context = build_preview(image, max_size=128)
        assert context["source_level"] == 0 and context["nonfinite_source_samples_omitted"] == 1


@pytest.fixture
def vision_double(monkeypatch):
    """PNG and LLM boundaries replaced; no claim to run either optional/native adapter."""
    state = {"calls": [], "connections": [], "selections": [], "pixels": [], "response": response()}

    def encode(pixels):
        state["pixels"].append(pixels.copy())
        return b"synthetic-attachment-boundary"

    monkeypatch.setitem(sys.modules, "imagecodecs", SimpleNamespace(png_encode=encode))
    monkeypatch.setitem(sys.modules, "llm", SimpleNamespace(Attachment=lambda **kw: SimpleNamespace(**kw)))

    def prompt(text, **kwargs):
        assert state["active"]
        state["calls"].append((json.loads(text), kwargs))
        def consume():
            assert state["active"]
            return state["response"]
        return SimpleNamespace(text=consume)

    @contextmanager
    def connect(**kwargs):
        state["connections"].append(kwargs)
        state["active"] = True
        try:
            yield SimpleNamespace(prompt=prompt)
        finally:
            state["active"] = False

    def source(name, *, allowed_scopes):
        state["selections"].append((name, allowed_scopes))
        return SimpleNamespace(name="work", default_model="vision", scope="institutional",
                               organization="test", connect=connect)

    state["registry"] = SimpleNamespace(source=source)
    monkeypatch.setattr("omeify.intelligence._load_registry", lambda: state["registry"])
    return state


def test_single_vision_request_and_crop_compatible_output(image_factory, vision_double):
    image = image_factory(np.arange(3 * 129 * 257, dtype=np.uint16).reshape(3, 129, 257),
                          channel_names=("CD3", "DAPI", "CD20"))
    doc = locate_regions(image, "right tissue", preview_size=128, registry=vision_double["registry"])
    assert len(vision_double["calls"]) == 1
    sent, options = vision_double["calls"][0]
    assert set(sent) == {"question", "context", "roi_size_px"}
    assert sent["context"]["base_width"] == 257 and sent["context"]["channel_index"] == 1
    assert vision_double["connections"][0]["requires"] == {"vision", "json_schema", "system_prompt"}
    assert vision_double["selections"] == [(None, ("institutional", "local"))]
    assert len(options["attachments"]) == 1 and options["attachments"][0].type == "image/png"
    assert "tools" not in options and options["stream"] is False
    assert "maxItems" not in json.dumps(options["schema"])  # Compact grammar; validated locally.
    assert doc["omeify"]["image_size"] == [257, 129] and len(parse_regions(doc)) == 1
    assert not vision_double["active"]
    vision_double["response"] = response((900, 0, 100, 500))
    with pytest.raises(IntelligenceError):
        locate_regions(image, "right tissue", preview_size=128)
    assert len(vision_double["calls"]) == 2  # No automatic retries.


def test_cli_visual_stdout_and_failure_does_not_replace_report(tmp_path, vision_double):
    path = tmp_path / "PRIVATE-NAME.ome.tiff"
    tifffile.imwrite(path, np.zeros((128, 256), dtype=np.uint16), ome=True,
                     metadata={"axes": "YX", "Channel": {"Name": ["DAPI"]}})
    args = ["inspect", str(path), "-i", "-q", "right tissue", "--geojson", "--preview-size", "128"]
    result = runner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["type"] == "FeatureCollection"
    assert "Sheetbend" in result.stderr and "PRIVATE-NAME" not in json.dumps(vision_double["calls"][0][0])
    report = tmp_path / "prior.json"
    report.write_text("keep")
    vision_double["response"] = "not JSON"
    result = runner().invoke(main, [*args, "-o", str(report)])
    assert result.exit_code != 0 and report.read_text() == "keep"
    assert result.stdout == ""
    for flags in [["--geojson"], ["-i", "--geojson"], ["--preview-size", "128"],
                  [*args[2:], "--intelligence-max-chars", "64000"]]:
        assert runner().invoke(main, ["inspect", str(path), *flags]).exit_code == 2


def test_rgb_display_and_bounded_channel_context(image_factory):
    from omeify import RGBImage

    rgb = np.empty((151, 263, 3), dtype=np.uint8)
    rgb[:] = [195, 99, 210]
    with RGBImage.from_array(rgb) as image:
        preview, context = build_preview(image, max_size=128)
        np.testing.assert_array_equal(preview, np.broadcast_to(rgb[0, 0], preview.shape))
        assert context["channel_selection"] == "rgb" and context["contrast"] is None
        with pytest.raises(ValueError, match="scalar-only"):
            build_preview(image, display_range=(0, 255))
    image = image_factory(np.zeros((16, 16), dtype=np.uint8), channel_names=("A" * 300,))
    _, context = build_preview(image, max_size=128)
    assert len(context["channel_name"]) == 256 and context["channel_name_truncated"]


def test_bad_request_settings_fail_before_rendering(image_factory, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Must not render a preview for an invalid request")

    monkeypatch.setattr("omeify.visual_intelligence.build_preview", fail)
    image = image_factory(np.zeros((16, 16), dtype=np.uint8))
    for options in [{"max_output_tokens": 0}, {"allowed_scopes": "local"},
                    {"roi_size": (32, 32)}, {"allowed_scopes": {"local", "institutional"}}]:
        with pytest.raises((ValueError, TypeError)):
            locate_regions(image, "right tissue", **options)


def test_real_vision_attachment_adapter(tmp_path, monkeypatch):
    """Actual optional transport/PNG serialization; the server is NOT a model."""
    import base64
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    sheetbend = pytest.importorskip("sheetbend", reason="Optional Sheetbend is not installed")
    pytest.importorskip("llm", minversion="0.35")
    pytest.importorskip("openai", minversion="3")
    codecs = pytest.importorskip("imagecodecs")
    from omeify import RGBImage

    monkeypatch.setenv("SHEETBEND_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            encoded = json.dumps({
                "id": "fixture", "object": "chat.completion", "created": 1, "model": "fixture",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response()},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = sheetbend.Registry.from_dict({
            "schema_version": 2, "sources": {"fixture": {
                "protocol": "openai-compatible",
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "scope": "local", "auth": {"type": "none"}, "default_model": "fixture",
                "models": {"fixture": {"capabilities": {
                    "vision": True, "json_schema": True, "system_prompt": True,
                }}},
            }},
        })
        rgb = np.full((173, 311, 3), [190, 105, 180], dtype=np.uint8)
        with RGBImage.from_array(rgb) as image:
            doc = locate_regions(image, "right tissue", registry=registry, preview_size=128,
                                 allowed_scopes=("local",))
        assert len(requests) == 1 and requests[0][0] == "/v1/chat/completions"
        body = requests[0][1]
        assert body["response_format"]["type"] == "json_schema" and "tools" not in body
        parts = body["messages"][-1]["content"]
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == 1
        url = images[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = codecs.png_decode(base64.b64decode(url.split(",", 1)[1], validate=True))
        assert decoded.shape == (71, 128, 3)
        np.testing.assert_array_equal(decoded, np.broadcast_to(rgb[0, 0], decoded.shape))
        payload = json.loads(next(p["text"] for p in parts if p["type"] == "text"))
        assert payload["context"]["base_width"] == 311
        assert doc["omeify"]["source"]["scope"] == "local"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
