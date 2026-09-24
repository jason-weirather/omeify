from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .configuration import WriterSettings
from .model import PreparedImage, WriteResult
from .pyramid import build_pyramid
from .writing import write_output

LOGGER = logging.getLogger(__name__)
VerificationCallback = Callable[[Path], dict[str, object]]


class WriterEngine:
    """Build pyramids, write, verify, and atomically install one OME-TIFF."""

    def __init__(self, settings: WriterSettings) -> None:
        if not isinstance(settings, WriterSettings):
            raise TypeError("settings must be WriterSettings")
        self.settings = settings

    def write(
        self,
        prepared: Sequence[PreparedImage],
        omexml: str,
        *,
        verify: VerificationCallback,
    ) -> WriteResult:
        """Run the shared file-construction lifecycle for prepared images."""

        images = tuple(prepared)
        if not images:
            raise ValueError("at least one prepared image is required")
        if not isinstance(omexml, str) or not omexml:
            raise ValueError("omexml must be a non-empty string")
        if not callable(verify):
            raise TypeError("verify must be callable")

        started = time.monotonic()
        self._prepare_destination()
        with tempfile.TemporaryDirectory(
            prefix="omeify-pyramid-",
            dir=(
                str(self.settings.cache_directory)
                if self.settings.cache_directory is not None
                else None
            ),
        ) as temporary_directory:
            temp_root = Path(temporary_directory)
            LOGGER.debug("Pyramid scratch directory: %s", temp_root)
            level_paths = self._build_pyramids(images, temp_root)
            temporary_output = self._temporary_output_path()
            try:
                if any(level_paths):
                    LOGGER.info(
                        "Writing final encoded OME-TIFF from base rasters and "
                        "staged pyramid levels"
                    )
                else:
                    LOGGER.info("Writing final encoded OME-TIFF from base raster(s)")
                write_started = time.monotonic()
                write_output(
                    images,
                    level_paths,
                    temporary_output,
                    omexml,
                    settings=self.settings,
                )
                LOGGER.info(
                    "Final encoded temporary output complete in %.2f seconds (%s bytes)",
                    time.monotonic() - write_started,
                    f"{temporary_output.stat().st_size:,}",
                )

                LOGGER.info(
                    "Verifying OME metadata, TIFF layout, decoding, and spot values"
                )
                verification_started = time.monotonic()
                verification = verify(temporary_output)
                LOGGER.info(
                    "Output verification passed in %.2f seconds",
                    time.monotonic() - verification_started,
                )

                LOGGER.info(
                    "Installing verified output atomically at %s",
                    self.settings.output_path,
                )
                output_size = temporary_output.stat().st_size
                if self.settings.overwrite:
                    os.replace(temporary_output, self.settings.output_path)
                else:
                    # Both paths are on the destination filesystem. Creating a
                    # hard link atomically fails if ANY entry already exists,
                    # including a dangling symlink or a competing writer's file.
                    # Do not fall back to check-then-replace or non-atomic copy.
                    os.link(temporary_output, self.settings.output_path)
                LOGGER.info("Verified output installed")
            finally:
                temporary_output.unlink(missing_ok=True)
        LOGGER.info("Temporary pyramid cache removed")
        LOGGER.info(
            "Writer complete in %.2f seconds; output size=%s bytes",
            time.monotonic() - started,
            f"{output_size:,}",
        )
        return WriteResult(
            verification=verification,
            output_size=output_size,
        )

    def _prepare_destination(self) -> None:
        if os.path.lexists(self.settings.output_path) and not self.settings.overwrite:
            raise FileExistsError(
                f"Output already exists: {self.settings.output_path}"
            )
        self.settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.cache_directory is not None:
            self.settings.cache_directory.mkdir(parents=True, exist_ok=True)

    def _build_pyramids(
        self,
        prepared: tuple[PreparedImage, ...],
        temp_root: Path,
    ) -> tuple[tuple[Path, ...], ...]:
        level_count = sum(len(item.level_shapes) - 1 for item in prepared)
        if level_count:
            LOGGER.info(
                "Building %s temporary uncompressed pyramid level(s) before "
                "final encoding",
                level_count,
            )
            started = time.monotonic()
        else:
            LOGGER.info("Pyramid construction skipped: base level(s) only")

        paths = tuple(
            build_pyramid(
                item,
                temp_root / f"image-{index}",
                tile_size=item.tile_size,
            )
            for index, item in enumerate(prepared)
        )
        if level_count:
            LOGGER.info(
                "Temporary pyramid construction complete in %.2f seconds",
                time.monotonic() - started,
            )
        return paths

    def _temporary_output_path(self) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".omeify-",
            suffix=".partial",
            dir=str(self.settings.output_path.parent),
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        temporary_output.unlink()
        return temporary_output
