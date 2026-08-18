from __future__ import annotations

from pathlib import Path

from omeify.utils.generic_conversion import GenericConversion


class AkoyaHEQptiff(GenericConversion):
    """Compatibility placeholder for the deliberately deferred RGB rewrite."""

    profile = "akoya_mif_qptiff"
    image_name = "WholeSlideHE"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        rename_channels: dict[str, str] | None = None,
    ) -> None:
        super().__init__(input_file_path, series=series, rename_channels=rename_channels)

    @property
    def input_type(self) -> str:
        return "Akoya H&E QPTIFF"

    def convert(self, *args, **kwargs):
        raise NotImplementedError(
            "The pure-Python H&E writer is intentionally deferred.  The next iteration should "
            "write one interleaved RGB image with SamplesPerPixel=3 instead of three grayscale "
            "channel pages."
        )
