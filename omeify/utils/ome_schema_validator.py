from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

LOGGER = logging.getLogger(__name__)


class OMESchemaValidator:
    """Validate OME-XML against a local OME 2016-06 schema.

    No network request is made.  The ``ome-schema`` package is preferred; a
    schema bundled beside tifffile is used as a fallback when available.
    """

    def __init__(self, schema_location: str | Path | None = None) -> None:
        self.schema_lxml: etree.XMLSchema | None = None
        self.schema_location: str | None = None

        if schema_location is None:
            try:
                from omeschema import get_ome_schema_path

                schema_location = get_ome_schema_path()
            except (ImportError, ModuleNotFoundError, AttributeError):
                schema_location = None

        if schema_location is None:
            try:
                import tifffile

                candidate = Path(tifffile.__file__).resolve().parent / "ome.xsd"
                if candidate.exists():
                    schema_location = candidate
            except (ImportError, OSError):
                schema_location = None

        if schema_location is not None:
            self.set_schema_lxml(schema_location)

    def set_schema_lxml(self, schema_location: str | Path) -> None:
        location = Path(schema_location)
        if not location.exists():
            raise FileNotFoundError(f"OME schema does not exist: {location}")
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        tree = etree.parse(str(location), parser)
        self.schema_lxml = etree.XMLSchema(tree)
        self.schema_location = str(location)

    def validate(self, xml_string: str) -> bool | None:
        if self.schema_lxml is None:
            LOGGER.warning(
                "OME-XML validation skipped because no local OME schema was available."
            )
            return None
        generated = etree.fromstring(xml_string.encode("utf-8"))
        valid = bool(self.schema_lxml.validate(generated))
        if not valid:
            for error in self.schema_lxml.error_log:
                LOGGER.warning("OME schema validation: %s", error)
        return valid
