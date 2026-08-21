from __future__ import annotations

from pathlib import Path

from omeify.io.akoya_qptiff import ChannelNameField
from omeify.io.channel import ChannelRenameMapping, RenameChannelsBy
from omeify.utils.generic_conversion import GenericConversion


class AkoyaFusionQPTiff(GenericConversion):
    """Convert planar multiplex Akoya Fusion QPTIFF to standardized OME-TIFF."""

    profile = "akoya_fusion_qptiff"
    image_name = "WholeSlideMIF"
    default_channel_name_field: ChannelNameField = "auto"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        rename_channels: ChannelRenameMapping | None = None,
        *,
        rename_channels_by: RenameChannelsBy | None = None,
        channel_name_field: ChannelNameField = "auto",
    ) -> None:
        super().__init__(
            input_file_path,
            series=series,
            rename_channels=rename_channels,
            rename_channels_by=rename_channels_by,
            channel_name_field=channel_name_field,
        )

    @property
    def input_type(self) -> str:
        return "Akoya Fusion multiplex QPTIFF"
