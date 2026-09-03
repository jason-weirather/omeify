from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .pixel_size import PixelSize, normalize_ome_pixel_size

ImageType = Literal["multichannel", "rgb", "label"]

_SUPPORTED_SCALAR_DTYPES = {
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
}


@dataclass(frozen=True)
class OMEImageSpec:
    """Normalized image description consumed by :class:`OMETiffWriter`.

    ``channel_names`` describes logical OME ``Channel`` elements. RGB therefore
    has one logical channel named ``RGB`` and three samples per pixel.
    """

    image_type: ImageType
    axes: str
    shape: tuple[int, ...]
    dtype: np.dtype
    channel_names: tuple[str, ...]
    pixel_size: PixelSize
    icc_profile: bytes | None = None
    significant_bits_override: int | None = None

    def __post_init__(self) -> None:
        image_type = str(self.image_type)
        if image_type not in {"multichannel", "rgb", "label"}:
            raise ValueError(
                "image_type must be 'multichannel', 'rgb', or 'label'; "
                f"found {self.image_type!r}"
            )
        object.__setattr__(self, "image_type", image_type)
        object.__setattr__(self, "axes", str(self.axes))
        object.__setattr__(self, "shape", tuple(int(item) for item in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype).newbyteorder("="))
        object.__setattr__(
            self,
            "channel_names",
            tuple(str(item) for item in self.channel_names),
        )
        object.__setattr__(self, "pixel_size", normalize_ome_pixel_size(self.pixel_size))
        if self.icc_profile is not None:
            object.__setattr__(self, "icc_profile", bytes(self.icc_profile))
        if self.significant_bits_override is not None:
            if (
                isinstance(self.significant_bits_override, bool)
                or not isinstance(self.significant_bits_override, int)
            ):
                raise TypeError("significant_bits_override must be an integer or None")
            storage_bits = int(self.dtype.itemsize * 8)
            if not 1 <= self.significant_bits_override <= storage_bits:
                raise ValueError(
                    "significant_bits_override must be between 1 and the dtype storage width "
                    f"({storage_bits})"
                )

        if len(self.axes) != len(self.shape):
            raise ValueError(f"Shape {self.shape} does not match axes {self.axes!r}")
        if any(item < 1 for item in self.shape):
            raise ValueError(f"OME image dimensions must be positive, found {self.shape}")
        if "Y" not in self.axes or "X" not in self.axes:
            raise ValueError(f"OME image axes must contain Y and X, found {self.axes!r}")
        if self.dtype.name not in _SUPPORTED_SCALAR_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_SCALAR_DTYPES))
            raise TypeError(
                f"Unsupported OME-TIFF dtype {self.dtype}. Supported dtypes are: {supported}."
            )
        if any(not name for name in self.channel_names):
            raise ValueError("OME logical channel names must be non-empty")

        if image_type == "rgb":
            if self.axes != "YXS" or self.shape[-1] != 3:
                raise ValueError(
                    "RGB OME-TIFF output requires axes='YXS' and shape (Y, X, 3); "
                    f"found axes={self.axes!r}, shape={self.shape}."
                )
            if self.dtype != np.dtype("uint8"):
                raise TypeError(f"RGB OME-TIFF output requires uint8 pixels, found {self.dtype}")
            if len(self.channel_names) != 1:
                raise ValueError(
                    "RGB OME metadata requires one logical channel name, normally 'RGB'"
                )
        elif image_type == "label":
            if self.axes != "YX" or len(self.shape) != 2:
                raise ValueError(
                    "Label OME-TIFF output requires one YX label raster; "
                    f"found axes={self.axes!r}, shape={self.shape}."
                )
            if not np.issubdtype(self.dtype, np.integer):
                raise TypeError(
                    f"Label OME-TIFF output requires an integer dtype, found {self.dtype}"
                )
            if len(self.channel_names) != 1:
                raise ValueError("Label OME-TIFF output requires exactly one logical channel name")
            if self.icc_profile is not None:
                raise ValueError("ICC profiles are only valid for RGB images")
        else:
            if self.axes not in {"YX", "CYX"}:
                raise ValueError(
                    "Multichannel OME-TIFF output currently requires axes='YX' or 'CYX'; "
                    f"found {self.axes!r}."
                )
            expected_channels = self.shape[0] if self.axes == "CYX" else 1
            if len(self.channel_names) != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} logical channel names for {self.axes} shape "
                    f"{self.shape}, found {len(self.channel_names)}."
                )
            if self.icc_profile is not None:
                raise ValueError("ICC profiles are only valid for RGB images")

    @classmethod
    def from_shape(
        cls,
        *,
        image_type: ImageType,
        axes: str,
        shape: Sequence[int],
        dtype: np.dtype | str | type,
        channel_names: Sequence[str] | None,
        pixel_size: PixelSize,
        icc_profile: bytes | None = None,
    ) -> "OMEImageSpec":
        normalized_shape = tuple(int(item) for item in shape)
        if channel_names is None:
            if image_type == "rgb":
                names = ("RGB",)
            elif image_type == "label":
                names = ("Labels",)
            else:
                count = normalized_shape[0] if axes == "CYX" else 1
                names = tuple(f"Channel {index + 1}" for index in range(count))
        else:
            names = tuple(str(item) for item in channel_names)
        return cls(
            image_type=image_type,
            axes=axes,
            shape=normalized_shape,
            dtype=np.dtype(dtype),
            channel_names=names,
            pixel_size=pixel_size,
            icc_profile=icc_profile,
        )

    @property
    def size_y(self) -> int:
        return int(self.shape[self.axes.index("Y")])

    @property
    def size_x(self) -> int:
        return int(self.shape[self.axes.index("X")])

    @property
    def samples_per_pixel(self) -> int:
        return 3 if self.is_rgb else 1

    @property
    def logical_channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def size_c(self) -> int:
        return self.logical_channel_count * self.samples_per_pixel

    @property
    def plane_count(self) -> int:
        return self.logical_channel_count

    @property
    def significant_bits(self) -> int:
        if self.significant_bits_override is not None:
            return self.significant_bits_override
        return int(self.dtype.itemsize * 8)

    @property
    def is_rgb(self) -> bool:
        return self.image_type == "rgb"

    @property
    def is_label(self) -> bool:
        return self.image_type == "label"

    @property
    def output_axes(self) -> str:
        return self.axes

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def shape_cyx(self) -> tuple[int, int, int]:
        return self.size_c, self.size_y, self.size_x

    @property
    def photometric(self) -> str:
        return "rgb" if self.is_rgb else "minisblack"
