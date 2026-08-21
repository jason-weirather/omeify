from __future__ import annotations

from typing import Any

from .miti_header_validator import MITIHeaderValidation, validate_miti_ome_tiff_header
from .ome_schema_validator import OMESchemaValidator
from omeify.io.tiff import TiffPlaneReader

from .tiff_image_features import (
    ImageMetadata,
    TiffImageFeatures,
    TiffImageSource,
    TiffMIFSource,
)

__all__ = [
    "GenericConversion",
    "ImageMetadata",
    "MITIHeaderValidation",
    "OMESchemaValidator",
    "TiffImageFeatures",
    "TiffImageSource",
    "TiffMIFSource",
    "TiffPlaneReader",
    "validate_miti_ome_tiff_header",
]


def __getattr__(name: str) -> Any:
    """Lazily expose GenericConversion without creating an import cycle."""

    if name == "GenericConversion":
        from .generic_conversion import GenericConversion

        return GenericConversion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
