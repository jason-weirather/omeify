from __future__ import annotations

import numpy as np

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import ArrayPlaneReader, PlaneReader


class ArrayPlanes:
    """Tiny independent array oracle for low-level writer/verification tests."""

    def __init__(self, array: np.ndarray, spec: OMEImageSpec) -> None:
        self._array = np.asarray(array)
        self._spec = spec

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        del cache_mib
        if self._spec.axes == "CYX":
            return [
                ArrayPlaneReader(self._array[index])
                for index in range(self._spec.plane_count)
            ]
        return [ArrayPlaneReader(self._array)]
