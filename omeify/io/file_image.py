"""File-origin metadata; image access itself is inherited from Image."""
from __future__ import annotations

from pathlib import Path
from typing import cast

from omeify.inspection import TiffInspector

from ._tiff_sources.base import TiffSource
from .base import Image


class FileImage(Image):
    """Private shared facade for format readers. No duplicate pixel-reading path."""

    @property
    def _file_source(self) -> TiffSource:
        return cast(TiffSource, self._source)

    @property
    def path(self) -> Path:
        return self._file_source.path

    @property
    def series_index(self) -> int:
        return self._file_source.series_index

    @property
    def series_names(self) -> tuple[str | None, ...]:
        self._ensure_open()
        return self._file_source.series_names()

    @property
    def series_name(self) -> str | None:
        self._ensure_open()
        return self._file_source.series_name()

    @property
    def native_axes(self) -> str:
        self._ensure_open()
        return str(self._file_source.native_levels[0].axes)

    @property
    def native_shape(self) -> tuple[int, ...]:
        self._ensure_open()
        return tuple(self._file_source.native_levels[0].shape)

    @property
    def source_byte_order(self) -> str:
        self._ensure_open()
        return "big" if self._file_source._tiff.byteorder == ">" else "little"

    def inspect(self) -> TiffInspector:
        """Return file diagnostics using the current handle; never read image pixels."""
        self._ensure_open()
        return self._file_source.inspect()
