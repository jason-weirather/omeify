"""Focused question mode: offline evidence, request, schema, CLI and terminal tests.

The endpoint is a model double. These tests check the application contract, not
whether a real language model interprets microscopy metadata correctly.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from importlib.resources import files
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from jsonschema import Draft202012Validator

from omeify import TiffInspector
from omeify import intelligence as ai
from omeify.cli import main
from omeify.inspection import _question_block


def write_image(path: Path, names: tuple[str, ...] = ("DAPI", "CD3", "PD-L1")) -> None:
    tifffile.imwrite(
        path, np.zeros((len(names), 16, 32), np.uint16), ome=True,
        photometric="minisblack", resolution=(20000, 20000), resolutionunit="CENTIMETER",
        metadata={
            "axes": "CYX", "Channel": {"Name": names},
            "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5, "PhysicalSizeYUnit": "µm",
        },
    )


def answer_for(packet: dict) -> dict:
    items = []
    for record in packet["records"]:
        for location in record["locations"]:
            match = re.search(r"/OME/Image\[(\d+)\]/Pixels\[0\]/Channel\[(\d+)\]/@Name$", location)
            if match:
                items.append((int(match[1]), int(match[2]), record["id"]))
    return {
        "status": "answered",
        "paragraphs": [{
            "text": "The declared channels, in metadata order, are:", "record_ids": [],
        }],
        "items": [{"label": f"Image {image}, channel {channel}", "record_id": record_id}
                  for image, channel, record_id in sorted(items)],
        "cautions": [],
    }


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "PRIVATE_CURRENT_FILENAME.ome.tif"
    write_image(path)
    return path


@pytest.fixture
def model_double(monkeypatch):
    state = {"calls": [], "selections": [], "connections": [], "active": False, "reply": None}

    def prompt(value, **kwargs):
        assert state["active"]
        state["calls"].append((json.loads(value), kwargs))

        def text():
            assert state["active"], "Response consumed outside the connection"
            reply = state["reply"]
            if isinstance(reply, Exception):
                raise reply
            if reply is not None:
                return reply if isinstance(reply, str) else json.dumps(reply)
            payload = json.loads(value)
            if "question" in payload:
                return json.dumps(answer_for(payload["metadata"]))
            record = payload["records"][0]
            return json.dumps({
                "overview": {"text": "The file contains metadata.", "record_ids": [record["id"]]},
                "findings": [], "cautions": [],
            })

        return SimpleNamespace(text=text)

    @contextmanager
    def connect(**kwargs):
        state["connections"].append(kwargs)
        state["active"] = True
        try:
            yield SimpleNamespace(prompt=prompt)
        finally:
            state["active"] = False

    source = SimpleNamespace(
        name="work", default_model="test-model", scope="institutional", organization="test",
        connect=connect,
    )

    def select(name, **kwargs):
        state["selections"].append((name, kwargs))
        return source

    state["registry"] = SimpleNamespace(source=select)
    monkeypatch.setattr(ai, "_load_registry", lambda: state["registry"])
    return state


@pytest.fixture
def runner():
    options = {"mix_stderr": False} if "mix_stderr" in signature(CliRunner).parameters else {}
    return CliRunner(**options)


@pytest.mark.parametrize("flag", ["--question", "-q"])
def test_question_requires_explicit_intelligence(image_path, model_double, runner, flag):
    result = runner.invoke(main, ["inspect", str(image_path), flag, "List channels"])
    assert result.exit_code == 2
    assert "requires --intelligence / -i" in result.stderr
    assert not model_double["calls"] and not model_double["selections"]


@pytest.mark.parametrize("question", ["", " ", "\n\t", "x" * 4097])
def test_invalid_cli_question_fails_before_collection(image_path, model_double, runner, monkeypatch,
                                                      question):
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not open a file for an invalid question")

    monkeypatch.setattr(tifffile, "TiffFile", forbidden)
    result = runner.invoke(main, ["inspect", str(image_path), "-i", "-q", question])
    assert result.exit_code == 2
    assert "Question failed JSON Schema validation" in result.stderr
    assert not model_double["selections"]


@pytest.mark.parametrize("flag", ["--question", "-q"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_question_replaces_summary_and_keeps_stdout_clean(image_path, model_double, runner,
                                                            flag, as_json):
    args = ["inspect", str(image_path), "-i", flag, "can u gib me a channel list"]
    result = runner.invoke(main, args + (["--json"] if as_json else []))
    assert result.exit_code == 0, result.stderr
    assert "Answering image question through Sheetbend" in result.stderr
    assert "Sheetbend..." not in result.stdout
    if as_json:
        report = json.loads(result.stdout)
        assert report["intelligence"]["question"] == "can u gib me a channel list"
        assert "summary" not in report["intelligence"]
        assert report["intelligence"]["schema_version"] == "1.2"
        assert report["intelligence"]["prompt_version"] == "3.0"
    else:
        plain = TiffInspector(image_path).render_text()
        assert result.stdout.startswith(plain + "\n\nImage question (advisory)")
        assert "Metadata intelligence (advisory)" not in result.stdout
        assert "Dates: not reported" not in result.stdout
        assert "  - Image 0, channel 0: DAPI" in result.stdout
    assert len(model_double["calls"]) == 1


def test_question_uses_same_connection_controls_and_separates_user_input(image_path, model_double):
    question = '  List channels in µm file; question mentions /user/supplied/path.  '
    inspector = TiffInspector(image_path)
    report = inspector.summarize_metadata(
        question=question, registry=model_double["registry"], source_name="chosen",
        model_name="alternate", allowed_scopes=["institutional"],
        max_metadata_chars=40000, max_output_tokens=16384,
    )
    payload, options = model_double["calls"][0]
    assert payload["question"] == question == report["question"]
    assert set(payload) == {"question", "metadata"}
    assert question not in options["system"]
    assert payload["metadata"]["coverage"]["max_metadata_chars"] == 40000
    assert options["options"] == {"max_tokens": 16384}
    assert options["stream"] is False
    assert "tools" not in options and "attachments" not in options
    assert model_double["selections"] == [("chosen", {"allowed_scopes": ("institutional",)})]
    assert model_double["connections"] == [{
        "model": "alternate", "requires": {"json_schema", "system_prompt"},
        "application": "omeify", "tool": "inspect",
    }]
    assert not model_double["active"]
    assert "UNTRUSTED DATA" in options["system"]
    assert "No raster pixels were read" in options["system"]
    assert "No Markdown" in options["system"]
    assert inspector.validation_errors() == ()


def test_cli_question_forwards_intelligence_options(image_path, model_double, runner):
    result = runner.invoke(main, [
        "inspect", str(image_path), "-i", "-q", "Channels?",
        "--intelligence-source", "chosen", "--intelligence-scope", "local",
        "--intelligence-max-chars", "40000", "--intelligence-max-output-tokens", "16384",
    ])
    assert result.exit_code == 0, result.stderr
    assert model_double["selections"] == [("chosen", {"allowed_scopes": ("local",)})]
    payload, options = model_double["calls"][0]
    assert payload["metadata"]["coverage"]["max_metadata_chars"] == 40000
    assert options["options"] == {"max_tokens": 16384}


def test_metadata_statistics_never_decode_or_send_current_paths(
    image_path, model_double, monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected pixel decoding")

    for cls in (tifffile.TiffFile, tifffile.TiffPage, tifffile.TiffPageSeries):
        monkeypatch.setattr(cls, "asarray", forbidden)
    report = TiffInspector(image_path).summarize_metadata(question="List the channels")
    packet = model_double["calls"][0][0]["metadata"]
    text = json.dumps(packet, ensure_ascii=False)
    assert str(image_path) not in text and image_path.name not in text
    stats = {r["locations"][0]: r["value"] for r in packet["records"]
             if r["locations"][0].startswith("omeify/inspection/")}
    file_stats = json.loads(stats["omeify/inspection/file"].split(": ", 1)[1])
    layout = json.loads(stats["omeify/inspection/series[0]"].split(": ", 1)[1])
    assert file_stats["size_bytes"] == image_path.stat().st_size
    assert file_stats["raster_pixels_examined"] is False
    assert layout["shape"] == [3, 16, 32] and layout["dtype"] == "uint16"
    assert layout["array_bytes"] == 3 * 16 * 32 * 2
    assert any(r["locations"][0].startswith("omeify/calibration/") for r in report["records"])


def test_question_evidence_is_independent_of_display_detail(image_path, model_double):
    first = TiffInspector(image_path, detail=0, max_text_length=1)
    second = TiffInspector(image_path, detail=3, max_text_length=None)
    a = first.summarize_metadata(question="Channels?")
    b = second.summarize_metadata(question="Channels?")
    assert a["records"] == b["records"] and a["coverage"] == b["coverage"]
    assert first.report["detail"] == 0 and second.report["detail"] == 3
    assert "PD-L1" in first.render_text().split("Image question (advisory)", 1)[1]


def test_question_record_budget_and_word_priority(tmp_path):
    path = tmp_path / "many-fields.tif"
    description = "<Vendor>" + "".join(
        f"<SampleID{i}>identifier-{i}</SampleID{i}>" for i in range(300)
    ) + "<ChannelName>LAST-MARKER</ChannelName></Vendor>"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8), description=description, metadata=None)
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        inspector = TiffInspector.from_tiff(tiff, file_path=path, detail=0)
        packet = ai.collect_metadata(
            tiff, max_chars=4096, question="List channels", inspection=inspector.report,
            calibration=inspector.report["calibration"],
        )
    assert "LAST-MARKER" in {r["value"] for r in packet["records"]}
    coverage = packet["coverage"]
    assert coverage["metadata_chars"] == len(json.dumps(
        packet["records"], ensure_ascii=False, separators=(",", ":"),
    )) <= 4096
    assert coverage["records_available"] > coverage["records_included"]
    assert len({r["id"] for r in packet["records"]}) == len(packet["records"])
    computed = [r for r in packet["records"] if r.get("origin") == "computed"]
    assert len(computed) == coverage["computed_records_included"]


def test_more_than_eight_channels_preserve_order_values_and_unicode(tmp_path, model_double):
    path = tmp_path / "channels.ome.tif"
    names = ("DAPI", "μ-chain", "µm-reference", "CD3", "CD3", *[f"marker-{i}" for i in range(15)])
    write_image(path, names)
    inspector = TiffInspector(path)
    report = inspector.summarize_metadata(question="List all channels")
    assert [item["value"] for item in report["answer"]["items"]] == list(names)
    catalog = {r["id"]: r for r in report["records"]}
    for item in report["answer"]["items"]:
        evidence = item["evidence"][0]
        assert item["value"] == evidence["quote"] == catalog[evidence["record_id"]]["value"]
    assert inspector.validation_errors() == ()
    assert len(model_double["calls"]) == 1
    assert "full citations in --json" in inspector.render_text()


def test_multiple_images_are_not_merged(tmp_path, model_double):
    path = tmp_path / "two-images.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        for names in (("DAPI", "CD3"), ("DAPI", "CD8")):
            writer.write(
                np.zeros((2, 16, 32), np.uint8), photometric="minisblack",
                metadata={"axes": "CYX", "Channel": {"Name": names}},
            )
    result = TiffInspector(path).summarize_metadata(question="List channels from each image")
    assert [item["value"] for item in result["answer"]["items"]] == ["DAPI", "CD3", "DAPI", "CD8"]
    assert [item["label"] for item in result["answer"]["items"]] == [
        "Image 0, channel 0", "Image 0, channel 1", "Image 1, channel 0", "Image 1, channel 1",
    ]


@pytest.mark.parametrize("question", ["", "\n\t ", "x" * 4097, 1, True, ["Channels?"]])
def test_bad_api_question_is_rejected_before_registry_access(model_double, question):
    with pytest.raises(ai.IntelligenceError, match="Question failed JSON Schema"):
        ai.summarize_metadata({}, question=question)
    assert not model_double["selections"]


def test_question_schema_binds_all_references_and_retains_local_constraints():
    schema = ai.metadata_question_schema(record_ids=["m8", "m23"])
    props = schema["properties"]
    for selector in (
        props["paragraphs"]["items"]["properties"]["record_ids"]["items"],
        props["items"]["items"]["properties"]["record_id"],
        props["cautions"]["items"]["properties"]["record_ids"]["items"],
    ):
        assert selector == {"type": "string", "enum": ["m8", "m23"]}
    for keyword in ('"$ref"', '"$defs"', '"maxLength"', '"pattern"', '"maxItems"'):
        assert keyword not in json.dumps(schema)
    packaged = json.loads(files("omeify.schemas").joinpath("metadata_intelligence.schema.json")
                          .read_text())
    Draft202012Validator.check_schema(packaged)
    assert props["status"]["enum"] == packaged["$defs"]["answer_status"]["enum"]
    assert packaged["$defs"]["question"]["maxLength"] == 4096


@pytest.mark.parametrize("field", ["paragraphs", "items", "cautions"])
def test_unknown_answer_record_fails_safely(image_path, model_double, field):
    reply = {"status": "answered", "paragraphs": [{"text": "PRIVATE_PROSE", "record_ids": []}],
             "items": [], "cautions": []}
    if field == "items":
        reply[field] = [{"label": "PRIVATE_LABEL", "record_id": "m999999"}]
    else:
        reply[field] = [{"text": "PRIVATE_PROSE", "record_ids": ["m999999"]}]
    model_double["reply"] = reply
    with pytest.raises(ai.IntelligenceError) as error:
        TiffInspector(image_path).summarize_metadata(question="Channels?")
    assert f"$.{field}[0].record_id" in str(error.value)
    assert "PRIVATE" not in str(error.value)
    assert len(model_double["calls"]) == 1


@pytest.mark.parametrize("reply", [
    "not JSON", '```json\n{}\n```', '{"a":1,"a":2}', '{"bad":NaN}', "[]", "{}", " ",
])
def test_question_invalid_json_never_becomes_empty_success(image_path, model_double, reply):
    model_double["reply"] = reply
    with pytest.raises(ai.IntelligenceError):
        TiffInspector(image_path).summarize_metadata(question="Channels?")
    assert len(model_double["calls"]) == 1 and not model_double["active"]


@pytest.mark.parametrize("text", [" ", "x" * 2001])
def test_question_full_constraints_are_applied_locally(image_path, model_double, text):
    model_double["reply"] = {
        "status": "unavailable", "paragraphs": [{"text": text, "record_ids": []}],
        "items": [], "cautions": [],
    }
    assert Draft202012Validator(ai.metadata_question_schema()).is_valid(model_double["reply"])
    with pytest.raises(ai.IntelligenceError, match="paragraphs"):
        TiffInspector(image_path).summarize_metadata(question="Mean intensity?")


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_non_unavailable_answer_requires_support(image_path, model_double, status):
    model_double["reply"] = {
        "status": status, "paragraphs": [{"text": "Made-up measurement.", "record_ids": []}],
        "items": [], "cautions": [],
    }
    with pytest.raises(ai.IntelligenceError, match="must cite supporting records"):
        TiffInspector(image_path).summarize_metadata(question="Mean intensity?")


def test_unavailable_pixel_question_can_honestly_have_no_evidence(image_path, model_double):
    model_double["reply"] = {
        "status": "unavailable",
        "paragraphs": [{"text": "No pixel-intensity measurements are available in this packet.",
                        "record_ids": []}],
        "items": [], "cautions": [],
    }
    inspector = TiffInspector(image_path)
    result = inspector.summarize_metadata(question="What is the mean DAPI intensity?")
    assert result["answer"]["status"] == "unavailable"
    assert inspector.validation_errors() == ()
    assert "Answer (unavailable):" in inspector.render_text()


def test_renderer_wraps_and_escapes_controls_without_altering_json(image_path, model_double,
                                                                monkeypatch):
    import omeify.inspection as inspection

    monkeypatch.setattr(
        inspection.shutil, "get_terminal_size", lambda **kw: SimpleNamespace(columns=52),
    )
    inspector = TiffInspector(image_path)
    result = inspector.summarize_metadata(question="Channels?\x1b[2J\r\b\u202e")
    result["answer"]["paragraphs"][0]["text"] = (
        "Long answer paragraph. " * 25 + "\x1b[31m\r\b\u202e"
    )
    result["source"]["name"] = "host\x1b[2J"
    block = _question_block(result)
    assert all(len(line) <= 52 for line in block.splitlines())
    for unsafe in ("\x1b", "\r", "\b", "\u202e"):
        assert unsafe not in block
    assert "\\x1b" in block and "\\u202e" in block
    stored = json.loads(inspector.to_json())["intelligence"]
    assert "\x1b" in stored["question"] and "\x1b" in stored["answer"]["paragraphs"][0]["text"]
    assert inspector.validation_errors() == ()
    assert len(model_double["calls"]) == 1


def test_question_and_summary_replace_each_other_without_render_requests(image_path, model_double):
    inspector = TiffInspector(image_path)
    before = deepcopy(inspector.report)
    inspector.summarize_metadata(question="Channels?")
    assert "answer" in inspector.report["intelligence"]
    assert "summary" not in inspector.report["intelligence"]
    assert {k: v for k, v in inspector.report.items() if k != "intelligence"} == before
    inspector.to_json()
    inspector.render_text()
    inspector._repr_html_()
    assert len(model_double["calls"]) == 1
    inspector.summarize_metadata()
    assert "summary" in inspector.report["intelligence"]
    assert "answer" not in inspector.report["intelligence"]
    assert "question" not in model_double["calls"][1][0]
    inspector.summarize_metadata(question="Channels?")
    assert "Image question" in inspector.render_text()
    assert "Metadata intelligence" not in inspector.render_text()
    assert len(model_double["calls"]) == 3


def test_schema_rejects_simultaneous_answer_and_summary(image_path, model_double):
    inspector = TiffInspector(image_path)
    result = inspector.summarize_metadata(question="Channels?")
    result["summary"] = {
        "overview": {"text": "A summary", "evidence": result["answer"]["items"][0]["evidence"]},
        "findings": [], "cautions": [],
    }
    assert inspector.validation_errors()


def test_question_failure_preserves_existing_output(image_path, model_double, runner, tmp_path):
    output = tmp_path / "keep.json"
    output.write_text("original report")
    before = image_path.read_bytes()
    model_double["reply"] = RuntimeError("PRIVATE_PROVIDER_BODY")
    result = runner.invoke(main, [
        "inspect", str(image_path), "-i", "-q", "Channels?", "-o", str(output),
    ])
    assert result.exit_code == 1 and result.stdout == ""
    assert "PRIVATE_PROVIDER_BODY" not in result.stderr
    assert output.read_text() == "original report" and image_path.read_bytes() == before
    assert len(model_double["calls"]) == 1


def test_question_output_file_and_borrowed_handle(image_path, model_double, runner, tmp_path):
    output = tmp_path / "answer.txt"
    result = runner.invoke(main, [
        "inspect", str(image_path), "-i", "-q", "Channels?", "-o", str(output),
    ])
    assert result.exit_code == 0 and result.stdout == ""
    assert "Image question" in output.read_text()
    with tifffile.TiffFile(image_path, _multifile=False) as tiff:
        inspector = TiffInspector.from_tiff(tiff, file_path=image_path)
        inspector.summarize_metadata(question="Channels?")
        assert not tiff.filehandle.closed
        assert inspector.validation_errors() == ()


@pytest.mark.parametrize("link", [False, True])
def test_question_cannot_write_report_over_image(image_path, model_double, runner, tmp_path, link):
    output = image_path
    if link:
        output = tmp_path / "hardlink.json"
        output.hardlink_to(image_path)
    before = image_path.read_bytes()
    result = runner.invoke(main, [
        "inspect", str(image_path), "-i", "-q", "Channels?", "-o", str(output),
    ])
    assert result.exit_code == 2
    assert image_path.read_bytes() == before
    assert not model_double["selections"]


@pytest.mark.parametrize("command", ["convert", "mutate", "version"])
def test_question_is_only_an_inspect_option(command, runner):
    result = runner.invoke(main, [command, "--question", "Channels?"])
    assert result.exit_code == 2
    assert "No such option: --question" in result.stderr
