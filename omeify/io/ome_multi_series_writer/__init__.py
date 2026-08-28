"""Named heterogeneous OME-TIFF image-series writing."""

from .image_series import OMEImageSeries
from .ome_multi_series_writer import OMEMultiSeriesWriter

__all__ = (
    "OMEImageSeries",
    "OMEMultiSeriesWriter",
)
