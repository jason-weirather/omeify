from omeify.converters import Bioformats2RawConverter
from omeify.converters import Raw2OmeTiffConverter
from omeify.utils.generate_ome_xml import generate_ome_xml

import os
import logging

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

    def convert(self, output_path):
        from omeify.utils import OMESchemaValidator
        
        # Convert the currently selected series
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Converting Series [{self.series}] into OME-TIFF")
        b2r_converter = Bioformats2RawConverter(self.input_file_path)
        zarr = b2r_converter.convert(series=self.series)
        tf = self.generate_original_tiff_features()
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Constructing OME metadata...")
        omexml = generate_ome_xml(tf, zarr)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Constructed OME metadata:\n{omexml}")
        # now replace the METADATA.ome.xml
        with open(os.path.join(zarr.store.path,'OME','METADATA.ome.xml'),'w') as output_file:
            output_file.write(omexml)
        self.raw2ometiff(zarr,output_path)
        b2r_converter.cleanup()
        osv = OMESchemaValidator()
        return {
            'output_path':output_path,
            'ome_xml':omexml,
            'ome_schema_location':osv.schema_location,
            'ome_xml_is_valid':osv.validate(omexml)
        }

