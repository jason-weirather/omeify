from __future__ import annotations

from pathlib import Path

from omeify.io.channel import ChannelRenameMapping, RenameChannelsBy
from omeify.utils.generic_conversion import GenericConversion


class AperioSVS(GenericConversion):
    """Convert an Aperio SVS baseline image as one interleaved RGB image."""

    profile = "svs"
    image_name = "WholeSlideHE"
    default_compression = "JPEG"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
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

    @property
    def input_type(self) -> str:
        return "Aperio SVS"
