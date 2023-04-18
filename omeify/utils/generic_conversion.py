from omeify.converters import Bioformats2RawConverter
from omeify.converters import Raw2OmeTiffConverter
from omeify.utils.generate_ome_xml import generate_ome_xml

import os
import logging
import hashlib

class GenericConversion:
    def __init__(self, input_file_path, series=None):
        self.input_file_path = input_file_path
        self.series = series
        self.logger = logging.getLogger(__name__)

    def raw2ometiff(self,zarr,output_path):
        #Raw2OmeTiffConverter(zarr.store.path).convert(output_path)
        raise NotImplementedError("Needs to be implemented for the specific input type.")

    def generate_original_tiff_features(self):
        # Implement the generic OME-TIFF metadata update step here
        # Will use self.input_file_path and self.series to generate ome tiff features
        # This may involve using the specific ImageFeatures class to parse the input
        raise NotImplementedError("Needs to be implemented for the specific input type.")

    def convert(self, output_path, display_uuid = True):
        from omeify.utils import OMESchemaValidator
        from omeschema import get_ome_schema_path
        from omeify import __version__ as my_omeify_version
        from tiffinspector import __version__ as my_tiffinspector_version

        # Convert the currently selected series
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Converting Series [{self.series}] into OME-TIFF")
        b2r_converter = Bioformats2RawConverter(self.input_file_path)
        zarr = b2r_converter.convert(series=self.series)
        tf = self.generate_original_tiff_features()
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Constructing OME metadata...")
        _d = generate_ome_xml(tf, zarr, display_uuid = display_uuid)
        omexml = _d['xml_string']
        myuuid = _d['uuid']
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Constructed OME metadata:\n{omexml}")
        # now replace the METADATA.ome.xml
        with open(os.path.join(zarr.store.path,'OME','METADATA.ome.xml'),'w') as output_file:
            output_file.write(omexml)
        self.raw2ometiff(zarr,output_path)
        b2r_converter.cleanup()
        osv = OMESchemaValidator(schema_location = get_ome_schema_path())
        return {
            'input_path':self.input_file_path,
            'input_md5_checksum': md5_checksum(self.input_file_path),
            'input_series':self.series,
            'output_path':output_path,
            'output_md5_checksum': md5_checksum(output_path),
            'output_uuid':myuuid,
            'ome_xml':omexml,
            'ome_schema_location':osv.schema_location,
            'ome_xml_is_valid':osv.validate(omexml),
            'versions':{
                'omeify':my_omeify_version,
                'bioformats2raw':Bioformats2RawConverter.get_version(),
                'raw2ometiff':Raw2OmeTiffConverter.get_version(),
                'tiff-inspector':my_tiffinspector_version,
            }
        }

def md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
