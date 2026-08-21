from __future__ import annotations

from pathlib import Path

from omeify.io.channel import ChannelRenameMapping, RenameChannelsBy
from omeify.io.pixel_size import PixelSize
from omeify.utils.generic_conversion import GenericConversion


class AkoyaComponentTiff(GenericConversion):
    profile = "component"
    image_name = "Component"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        pixel_size: PixelSize = PixelSize(0.496, 0.496, "µm"),
        rename_channels: ChannelRenameMapping | None = None,
        *,
        rename_channels_by: RenameChannelsBy | None = None,
    ) -> None:
        super().__init__(
            input_file_path,
            series=series,
            rename_channels=rename_channels,
            rename_channels_by=rename_channels_by,
        )
        self._pixel_size_override = pixel_size

    @property
    def input_type(self) -> str:
        return "Akoya Component TIFF"
