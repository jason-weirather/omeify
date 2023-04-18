import re, requests
from lxml import etree
import logging

class OMESchemaValidator:
    def __init__(self,
        schema_location = None,
        schema_reference_url='http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
    ):
        self.schema_lxml = None
        if schema_location is None:
            schema_location = schema_reference_url
        self.set_schema_lxml(schema_location)
        self.logger = logging.getLogger(__name__)

    def set_schema_lxml(self, schema_location):
        if re.match('https://',schema_location) or re.match('http:',schema_location):
            response = requests.get(schema_location)
            if response.status_code != 200:
                raise Exception(f"Error downloading schema: {response.status_code} {resonse.text}")
            self.schema_lxml = etree.XMLSchema(etree.fromstring(response.content))
        else:
            with open(schema_location) as inf:
                self.schema_lxml = etree.XMLSchema(etree.fromstring(inf.read()))
        self.schema_location = schema_location

    def validate(self, xml_string):
        generated_etree = etree.fromstring(xml_string.encode("utf-8"))
        
        check = self.schema_lxml.validate(generated_etree)
        if not check:
            try:
                self.schema_lxml.assertValid(generated_etree)
            except etree.DocumentInvalid as e:
                self.logger.warning(f"Validation failed with errors:\n{e}")
        return check

