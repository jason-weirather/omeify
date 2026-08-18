from __future__ import annotations

from pathlib import Path

from omeify.utils.generic_conversion import GenericConversion


class AkoyaComponentTiff(GenericConversion):
    profile = "component"
    image_name = "Component"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        physical_size_x_um: float = 0.496,
        physical_size_y_um: float = 0.496,
        rename_channels: dict[str, str] | None = None,
    ) -> None:
        super().__init__(input_file_path, series=series, rename_channels=rename_channels)
        self._physical_size_x_um = float(physical_size_x_um)
        self._physical_size_y_um = float(physical_size_y_um)

    @property
    def input_type(self) -> str:
        return "Akoya Component TIFF"
