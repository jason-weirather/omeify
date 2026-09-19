"""Index validation shared by images and providers."""
from __future__ import annotations
from collections.abc import Sequence
from numbers import Integral
import numpy as np

def normalize_channel_indices(
    selection: int | Sequence[int] | None,
    size: int,
    *,
    field: str = "channels",
) -> list[int]:
    if selection is None:
        return list(range(size))
    if isinstance(selection, (bool, np.bool_, str, bytes)):
        raise TypeError(f"{field} must be an integer index or a sequence of integer indices")
    if isinstance(selection, Integral):
        selected = [int(selection)]
    else:
        values = tuple(selection)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
            raise TypeError(f"{field} must contain integer indices, not booleans or floats")
        selected = [int(v) for v in values]
    if not selected:
        raise ValueError(f"{field} must contain at least one index")
    for index in selected:
        if index < 0 or index >= size:
            raise IndexError(f"Channel {index} is outside the available range 0..{size - 1}")
    return selected
