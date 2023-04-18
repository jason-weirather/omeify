import re, requests
from lxml import etree

class OMESchemaValidator:
    def __init__(self,
        schema_location = None,
        schema_reference_url='http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
    ):
        schema_lxml = None
        if schema_location is None:
            schema_location = schema_reference_url
        self.set_schema_lxml(schema_location)

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
        
        return self.schema_lxml.validate(generated_etree)

