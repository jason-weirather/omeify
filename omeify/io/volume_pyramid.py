"""Bounded XYZ pyramid arithmetic, without changing the established XY reducer.

Only spatial axes may be reduced. Time, channels and RGB samples are never
pooled. Mean levels describe a nominal regular center grid; an incomplete edge
bin uses its actual contributor count, just like the existing XY writer.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ._writer.pyramid import mean_downsample_2x
from .image_metadata import integer


@dataclass(frozen=True, slots=True)
class VolumePyramid:
    shapes: tuple[tuple[int, int, int], ...]
    steps: tuple[tuple[int, int, int], ...]


def plan_volume_pyramid(shape, *, tile_size=512, chunk_z=1, levels=None,
                        axes="xy", spacing=None, factors=None):
    """Plan native ZYX shapes and per-level integer factors, without reading pixels.

    ``axes='auto'`` is anisotropy-aware and requires physical Z/Y/X spacing.
    ``factors`` is an explicit sequence of ZYX 1/2 reductions and is mutually
    exclusive with levels. Automatic plans stop when the requested storage block
    fits, or fail at 32 reductions rather than silently producing a partial plan.
    """
    shape = tuple(integer(n, "volume dimension", minimum=1) for n in shape)
    if len(shape) != 3:
        raise ValueError("Volume shape must be ZYX")
    tile_size = integer(tile_size, "tile_size", minimum=1)
    chunk_z = integer(chunk_z, "chunk_z", minimum=1)
    if axes not in {"xy", "xyz", "auto"}:
        raise ValueError("pyramid_axes must be xy, xyz, or auto")
    if levels is not None:
        levels = integer(levels, "pyramid_levels")
        if levels > 32:
            raise ValueError("At most 32 pyramid reductions are supported")
    if spacing is not None:
        spacing = tuple(float(v) for v in spacing)
        if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
            raise ValueError("spacing must contain positive finite Z/Y/X sampling")
    if axes == "auto" and spacing is None:
        raise ValueError("Anisotropy-aware pyramids require calibrated Z/Y/X spacing")
    explicit = None
    if factors is not None:
        if levels is not None:
            raise ValueError("Choose explicit pyramid factors or a level count, not both")
        explicit = []
        for factor in factors:
            if len(explicit) == 32:
                raise ValueError("At most 32 pyramid reductions are supported")
            factor = tuple(integer(v, "reduction factor", minimum=1) for v in factor)
            if len(factor) != 3 or any(v not in {1, 2} for v in factor) or factor == (1, 1, 1):
                raise ValueError("Each ZYX reduction must contain 1/2 and reduce an axis")
            if axes == "xy" and factor[0] != 1:
                raise ValueError("XY pyramids cannot reduce Z")
            explicit.append(factor)
    shapes, steps = [shape], []
    sampling = spacing
    limit = len(explicit) if explicit is not None else levels if levels is not None else 32
    for i in range(limit):
        z, y, x = shapes[-1]
        if explicit is not None:
            step = explicit[i]
            # An explicit factor of two on a singleton axis would alter its
            # advertised sampling without actually reducing any pixels.
            if any(f == 2 and n == 1 for n, f in zip((z, y, x), step, strict=True)):
                raise ValueError("Explicit pyramid factor cannot reduce a singleton Z/Y/X axis")
        else:
            done_xy = y <= tile_size and x <= tile_size
            done_z = axes == "xy" or z <= chunk_z
            if levels is None and done_xy and done_z:
                break
            if axes == "xy":
                step = (1, 2 if y > 1 else 1, 2 if x > 1 else 1)
            elif axes == "xyz":
                step = (2 if z > 1 else 1, 2 if y > 1 else 1, 2 if x > 1 else 1)
            else:
                # Reduce the finest spatial sampling first. Z can never be
                # averaged merely because it has more planes than the XY grid.
                active_sampling = [s for n,s in zip((z,y,x),sampling,strict=True) if n > 1]
                if not active_sampling:
                    if levels is None:
                        break
                    raise ValueError("No spatial axis remains to reduce")
                smallest = min(active_sampling)
                step = tuple(2 if s < 2 * smallest and n > 1 else 1
                             for n, s in zip((z, y, x), sampling, strict=True))
                if step == (1, 1, 1):
                    if levels is None:
                        break
                    raise ValueError("No spatial axis remains to reduce")
        # Requested level counts must never create duplicate resolutions.
        # Auto plans have the same constraint, but can stop naturally.
        if step == (1, 1, 1):
            if levels is None and explicit is None:
                break
            raise ValueError("No spatial axis remains to reduce at the requested pyramid level")
        steps.append(step)
        shapes.append(tuple((n + f - 1) // f for n, f in zip(shapes[-1], step, strict=True)))
        if sampling is not None:
            sampling = tuple(s * f for s, f in zip(sampling, step, strict=True))
    if explicit is None and levels is None:
        z, y, x = shapes[-1]
        if y > tile_size or x > tile_size or (axes != "xy" and z > chunk_z):
            raise ValueError("Pyramid plan exceeds 32 reductions")
    return VolumePyramid(tuple(shapes), tuple(steps))


def reduce_zyx(block, factors, *, method="mean"):
    """Reduce one aligned native ZYX/ZYXS block, rounding integer means once.

    Caller requests blocks aligned to output bins. Factors are 1 or 2; only
    spatial dimensions are reduced. No plane-by-plane rounding precedes a Z mean.
    """
    block = np.asarray(block)
    factors = tuple(integer(v, "reduction factor", minimum=1) for v in factors)
    if (block.ndim not in (3, 4) or len(factors) != 3 or
            any(v not in (1, 2) for v in factors) or any(n < 1 for n in block.shape[:3])):
        raise ValueError("Expected a nonempty ZYX/ZYXS block and three factors of 1 or 2")
    if block.dtype.name not in {"uint8", "uint16", "uint32", "int8", "int16", "int32",
                               "float32", "float64"}:
        raise TypeError("Unsupported volume pixel type")
    if method not in {"mean", "nearest"}:
        raise ValueError("Reduction method must be mean or nearest")
    if method == "nearest" or factors == (1, 1, 1):
        return np.array(block[::factors[0], ::factors[1], ::factors[2]], copy=True, order="C")
    shape = tuple((n + f - 1) // f for n, f in zip(block.shape[:3], factors, strict=True))
    if factors == (1, 2, 2):
        # Existing 2D arithmetic is the authoritative XY policy, including its
        # floating accumulation order, signed-zero and finite-overflow behavior.
        return np.stack([mean_downsample_2x(plane, shape[1:]) for plane in block])
    full_shape = (*shape, *block.shape[3:])
    dtype = block.dtype
    accum_dtype = np.uint64 if dtype.kind == "u" else np.int64 if dtype.kind == "i" else np.float64
    total = np.zeros(full_shape, dtype=accum_dtype)
    counts = np.zeros(shape, dtype=np.uint8)
    offsets = tuple(itertools.product(*(range(f) for f in factors)))
    with np.errstate(over="ignore", invalid="ignore"):
        for offset in offsets:
            values = block[tuple(slice(o, None, f) for o, f in zip(offset, factors, strict=True))]
            target = tuple(slice(0, n) for n in values.shape[:3])
            total[target] += values.astype(accum_dtype, copy=False)
            counts[target] += 1
    divisor = counts if block.ndim == 3 else counts[..., None]
    if dtype.kind in "iu":
        quotient, remainder = np.divmod(total, divisor.astype(accum_dtype))
        # Floor division plus ties-to-even also works for negative signed sums.
        increment = (2 * remainder > divisor) | ((2 * remainder == divisor) & ((quotient & 1) != 0))
        return np.asarray(quotient + increment, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = total / divisor
    # Only float64 finite contributors can overflow a float64 sum of <=8
    # values. Float32 inputs cannot, even if their output contains many NaNs.
    # For float64, mask genuine NaN/Inf source groups in vectorized slices;
    # iterating over them in Python made NaN-heavy volumes extremely slow.
    if dtype.kind == "f" and dtype.itemsize == 8:
        overflow = ~np.isfinite(total)
        if np.any(overflow):
            for offset in offsets:
                values = block[tuple(slice(o, None, f) for o, f in zip(offset, factors, strict=True))]
                target = tuple(slice(0, n) for n in values.shape[:3])
                overflow[target] &= np.isfinite(values)
            # Only recover bins with ALL finite inputs whose running sum
            # overflowed. True source NaN/Inf follows the original IEEE rules.
            for position in np.argwhere(overflow):
                index = tuple(int(v) for v in position)
                window = tuple(slice(index[a]*f, min(block.shape[a], (index[a]+1)*f))
                               for a, f in enumerate(factors)) + index[3:]
                values = np.asarray(block[window], dtype=np.float64).reshape(-1)
                magnitude = float(np.max(np.abs(values)))
                if magnitude:
                    _, exponent = math.frexp(magnitude)
                    result[index] = np.ldexp(np.ldexp(values, -exponent).sum() / values.size, exponent)
    return np.asarray(result, dtype=dtype)
