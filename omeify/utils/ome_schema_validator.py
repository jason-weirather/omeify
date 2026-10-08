from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

LOGGER = logging.getLogger(__name__)


class OMESchemaValidator:
    """Validate OME-XML using the declared local XSD dependency or an explicit path.

    Construction fails if the XSD cannot be loaded. Validation never downloads a
    schema or reports a missing validator as a skipped/successful check.
    """

    def __init__(self, schema_location: str | Path | None = None) -> None:
        if schema_location is None:
            try:
                from omeschema import get_ome_schema_path
            except ImportError as exc:
                raise RuntimeError(
                    "OME-XML validation requires the ome-schema dependency; "
                    "install omeify's declared dependencies or supply schema_location."
                ) from exc
            schema_location = get_ome_schema_path()
        self.set_schema_lxml(schema_location)

    def set_schema_lxml(self, schema_location: str | Path) -> None:
        location = Path(schema_location)
        if not location.exists():
            raise FileNotFoundError(f"OME schema does not exist: {location}")
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        tree = etree.parse(str(location), parser)
        self.schema_lxml = etree.XMLSchema(tree)
        self.schema_location = str(location)

    def validate(self, xml_string: str) -> bool:
        generated = etree.fromstring(xml_string.encode("utf-8"))
        valid = bool(self.schema_lxml.validate(generated))
        if not valid:
            for error in self.schema_lxml.error_log:
                LOGGER.warning("OME schema validation: %s", error)
        return valid
