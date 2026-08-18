from __future__ import annotations

from pathlib import Path

from omeify.utils.generic_conversion import GenericConversion


class HaloMIFTiff(GenericConversion):
    profile = "halo_mif"
    image_name = "WholeSlideMIF"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        rename_channels: dict[str, str] | None = None,
    ) -> None:
        super().__init__(input_file_path, series=series, rename_channels=rename_channels)

    @property
    def input_type(self) -> str:
        return "HALO mIF TIFF"
