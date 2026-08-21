from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .tiff import PlaneReader

RenameChannelsBy = Literal["name", "index"]
ChannelRenameMapping = Mapping[str, str] | Mapping[int, str]


def normalize_channel_indices(
    selection: int | Sequence[int] | None,
    size: int,
    *,
    field: str = "channels",
) -> list[int]:
    if selection is None:
        return list(range(size))
    if isinstance(selection, bool):
        raise TypeError(f"{field} must be an integer index or a sequence of indices")
    selected = (
        [int(selection)]
        if isinstance(selection, Integral) and not isinstance(selection, bool)
        else [int(item) for item in selection]
    )
    if not selected:
        raise ValueError(f"{field} must contain at least one index")
    for index in selected:
        if index < 0 or index >= size:
            raise IndexError(f"Channel {index} is outside the available range 0..{size - 1}")
    return selected


def validate_channel_rename_mapping(
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[dict[str, str] | dict[int, str], RenameChannelsBy | None]:
    """Validate an explicit Python channel-rename mapping."""

    if not mapping:
        if by is not None and by not in {"name", "index"}:
            raise ValueError("rename_channels_by must be 'name' or 'index'")
        return {}, by
    if by not in {"name", "index"}:
        raise ValueError(
            "rename_channels_by must be explicitly set to 'name' or 'index' "
            "when rename_channels is provided"
        )
    values = list(mapping.values())
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise TypeError("Channel rename values must be non-empty strings")

    if by == "name":
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Name-based channel rename keys must be strings")
            if key.strip().lstrip("+").isdigit():
                raise ValueError(
                    f"Ambiguous name-based rename key {key!r}; integer-like keys belong "
                    "in rename_channels_by='index' mappings"
                )
            normalized[key] = value.strip()
        return normalized, by

    normalized_index: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("Index-based Python channel rename keys must be integers")
        index = int(key)
        if index < 0:
            raise ValueError("Index-based channel rename keys must be zero or greater")
        normalized_index[index] = value.strip()
    return normalized_index, by


def apply_channel_renames(
    channel_names: Sequence[str],
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[str, ...]:
    normalized, normalized_by = validate_channel_rename_mapping(mapping, by)
    names = tuple(str(item) for item in channel_names)
    if not normalized:
        return names

    if normalized_by == "name":
        name_mapping = normalized
        unknown = sorted(set(name_mapping) - set(names))
        if unknown:
            raise ValueError(
                "Name-based channel rename mapping contains unknown source names: "
                + ", ".join(repr(item) for item in unknown)
            )
        ambiguous = sorted(
            name for name in name_mapping if names.count(name) > 1
        )
        if ambiguous:
            raise ValueError(
                "Name-based channel rename mapping is ambiguous for duplicate source names: "
                + ", ".join(repr(item) for item in ambiguous)
                + "; use rename_channels_by='index'"
            )
        return tuple(name_mapping.get(name, name) for name in names)

    index_mapping = normalized
    out_of_range = sorted(index for index in index_mapping if index >= len(names))
    if out_of_range:
        raise IndexError(
            "Index-based channel rename mapping contains out-of-range indices: "
            + ", ".join(str(item) for item in out_of_range)
        )
    return tuple(index_mapping.get(index, name) for index, name in enumerate(names))


class Channel:
    """Lazy logical image channel backed by regional/random-access I/O.

    Metadata properties never read pixel data. ``array`` is the explicit escape
    hatch that materializes the full-resolution channel.
    """

    def __init__(
        self,
        *,
        index: int,
        channel_id: str,
        name: str,
        dtype: np.dtype | str | type,
        shape: Sequence[int],
        samples_per_pixel: int = 1,
        plane_reader_factory: Callable[[int], PlaneReader],
        array_reader: Callable[[int], np.ndarray] | None = None,
        ensure_available: Callable[[], None] | None = None,
        source_id: str | None = None,
        id_is_generated: bool = False,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.index = int(index)
        if self.index < 0:
            raise ValueError("Channel index must be zero or greater")
        self.id = str(channel_id)
        self.name = str(name)
        if not self.id:
            raise ValueError("Channel ID must be non-empty")
        if not self.name:
            raise ValueError("Channel name must be non-empty")
        self.dtype = np.dtype(dtype).newbyteorder("=")
        self.shape = tuple(int(item) for item in shape)
        if len(self.shape) not in {2, 3} or any(item < 1 for item in self.shape):
            raise ValueError(f"Channel shape must be positive YX or YXS, found {self.shape}")
        self.samples_per_pixel = int(samples_per_pixel)
        if self.samples_per_pixel < 1:
            raise ValueError("samples_per_pixel must be at least one")
        if len(self.shape) == 2 and self.samples_per_pixel != 1:
            raise ValueError("A YX Channel must have SamplesPerPixel=1")
        if len(self.shape) == 3 and self.shape[-1] != self.samples_per_pixel:
            raise ValueError(
                f"YXS Channel shape {self.shape} does not match "
                f"SamplesPerPixel={self.samples_per_pixel}"
            )
        self.source_id = None if source_id is None else str(source_id)
        self.id_is_generated = bool(id_is_generated)
        self.source_metadata = MappingProxyType(dict(source_metadata or {}))
        self._plane_reader_factory = plane_reader_factory
        self._array_reader = array_reader
        self._ensure_available = ensure_available
        self._readers: dict[int, PlaneReader] = {}

    @property
    def height(self) -> int:
        return int(self.shape[0])

    @property
    def width(self) -> int:
        return int(self.shape[1])

    def _ensure(self) -> None:
        if self._ensure_available is not None:
            self._ensure_available()

    def _reader(self, level: int = 0) -> PlaneReader:
        self._ensure()
        level_index = int(level)
        if level_index < 0:
            raise IndexError("Pyramid level must be zero or greater")
        reader = self._readers.get(level_index)
        if reader is None:
            reader = self._plane_reader_factory(level_index)
            self._readers[level_index] = reader
        return reader

    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        level: int = 0,
    ) -> np.ndarray:
        return np.ascontiguousarray(self._reader(level).read_region(y0, y1, x0, x1))

    def asarray(self, *, level: int = 0) -> np.ndarray:
        self._ensure()
        if self._array_reader is not None:
            value = np.asarray(self._array_reader(int(level)))
        else:
            reader = self._reader(level)
            value = reader.read_region(0, reader.height, 0, reader.width)
        if np.dtype(value.dtype).newbyteorder("=") != self.dtype:
            raise TypeError(
                f"Channel {self.index} materialized dtype {value.dtype}; expected {self.dtype}"
            )
        return np.ascontiguousarray(value)

    @property
    def array(self) -> np.ndarray:
        return self.asarray(level=0)

    def clear_cache(self) -> None:
        for reader in self._readers.values():
            reader.clear_cache()

    def __repr__(self) -> str:
        return (
            f"Channel(index={self.index}, id={self.id!r}, name={self.name!r}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )
