from ._version import __version__, get_version_info
from .inspection import TiffInspector
from .io import Image, LabelImage, MultichannelImage, OMETiffReader, RGBImage

__all__ = [
    "Image",
    "LabelImage",
    "MultichannelImage",
    "OMETiffReader",
    "RGBImage",
    "TiffInspector",
    "__version__",
    "get_version_info",
]
