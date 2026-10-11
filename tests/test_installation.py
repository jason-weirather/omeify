"""The Python baseline and complete image-I/O installation are public contracts."""
from __future__ import annotations

import ast
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Self, get_type_hints

import numpy as np
import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from omeify import Image, ImageSource, MultichannelImage, PixelSize
from omeify import _version

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_minimum_python_and_stdlib_toml(project):
    assert sys.version_info >= (3, 11), "Run the tests on a supported interpreter"
    supported = SpecifierSet(project["project"]["requires-python"])
    assert "3.10.99" not in supported
    assert "3.11.0" in supported
    assert project["tool"]["ruff"]["target-version"] == "py311"
    assert _version.tomllib is tomllib
    assert "Programming Language :: Python :: 3.10" not in project["project"]["classifiers"]


def test_complete_image_io_dependencies_are_unconditional(project):
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    for name in ("numpy", "zarr", "fsspec", "ome-schema"):
        selected = [r for r in requirements if r.name == name]
        assert len(selected) == 1, name
        assert selected[0].marker is None, name
    assert "http" in next(r for r in requirements if r.name == "fsspec").extras
    assert not any(r.name in {"tomli", "ome-zarr-models", "sheetbend"} for r in requirements)
    assert "zarr" not in project["project"].get("optional-dependencies", {})


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.13", "3.14"])
def test_supported_interpreters_select_exactly_one_tiff_codec_branch(project, python_version):
    environment = default_environment()
    environment.update(python_version=python_version, python_full_version=python_version + ".0")
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    active = [r for r in requirements if r.marker is None or r.marker.evaluate(environment)]
    for name in ("tifffile", "imagecodecs"):
        assert sum(r.name == name for r in active) == 1, (python_version, name)


def test_baseline_environment_definitions_and_selected_floors(project):
    assert (ROOT / "Dockerfile").read_text().splitlines()[0] == "FROM python:3.11-slim"
    assert "  - python=3.11" in (ROOT / "environment.yml").read_text().splitlines()
    environment = default_environment()
    environment.update(python_version="3.11", python_full_version="3.11.0")
    active = [Requirement(value) for value in project["project"]["dependencies"]]
    active = {r.name: r for r in active if r.marker is None or r.marker.evaluate(environment)}
    pins = {}
    for line in (ROOT / "constraints/python311-min.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            requirement = Requirement(line)
            specifiers = list(requirement.specifier)
            assert len(specifiers) == 1 and specifiers[0].operator == "=="
            pins[requirement.name] = specifiers[0].version
            assert specifiers[0].version in active[requirement.name].specifier
    assert pins["zarr"] == "3.1.3"
    assert pins["numpy"] == "1.26.4"


def test_all_source_test_and_example_syntax_supports_python311():
    # Grammar checks complement, but cannot replace, the real Python 3.11 test run.
    for directory in ("omeify", "tests", "examples"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))


@pytest.mark.parametrize("module", ["imagecodecs", "omeschema", "zarr", "fsspec", "aiohttp"])
def test_required_image_io_dependencies_import_without_skip(module):
    importlib.import_module(module)


def test_self_annotations_and_subclass_factories_preserve_lifetimes():
    class SpecializedImage(MultichannelImage):
        pass

    values = np.arange(30, dtype=np.uint16).reshape(5, 6)
    image = SpecializedImage.from_array(values, axes="YX", pixel_size=PixelSize(1, 1, "µm"))
    assert type(image) is SpecializedImage and not image.is_open
    with image as opened:
        assert opened is image
        handle = opened[0]
        with SpecializedImage.from_channels([handle]) as selected:
            assert type(selected) is SpecializedImage
            np.testing.assert_array_equal(selected[0].asarray(), values)
        # Cropped views deliberately return a semantic wrapper, not the reader subclass.
        with opened.crop(1, 4, 2, 5) as cropped:
            np.testing.assert_array_equal(cropped.asarray(), values[1:4, 2:5])
    with pytest.raises(RuntimeError):
        handle.asarray()
    for method in (Image.open, Image.__enter__, Image.from_array,
                   MultichannelImage.from_channels, ImageSource.open, ImageSource.__enter__,
                   PixelSize.from_tuple):
        assert get_type_hints(method)["return"] is Self
