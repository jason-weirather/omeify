from __future__ import annotations

import numpy as np

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import PlaneReader


class ArrayPlaneReader:
    """Random-access plane reader backed by a two-dimensional or YXS array."""

    def __init__(self, array: np.ndarray) -> None:
        value = np.asarray(array)
        if value.ndim == 2:
            samples_per_pixel = 1
        elif value.ndim == 3 and value.shape[-1] == 3:
            samples_per_pixel = 3
        else:
            raise ValueError(
                "ArrayPlaneReader requires a YX grayscale plane or a YXS RGB plane; "
                f"found shape {value.shape}."
            )
        self.array = value
        self.height = int(value.shape[0])
        self.width = int(value.shape[1])
        self.samples_per_pixel = samples_per_pixel
        self.dtype = np.dtype(value.dtype).newbyteorder("=")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.array.shape)

    def clear_cache(self) -> None:
        return None

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if not (0 <= y0 <= y1 <= self.height and 0 <= x0 <= x1 <= self.width):
            raise ValueError(
                f"Requested region {(y0, y1, x0, x1)} is outside plane shape {self.shape}"
            )
        return np.ascontiguousarray(self.array[y0:y1, x0:x1, ...])


class ArrayPlanes:
    """Tiny independent array oracle for low-level writer/verification tests."""

    def __init__(self, array: np.ndarray, spec: OMEImageSpec) -> None:
        self._array = np.asarray(array)
        self._spec = spec

    def plane_readers(self) -> list[PlaneReader]:
        if self._spec.axes == "CYX":
            return [
                ArrayPlaneReader(self._array[index])
                for index in range(self._spec.plane_count)
            ]
        return [ArrayPlaneReader(self._array)]
