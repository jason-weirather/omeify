from omeify.converters import Bioformats2RawConverter
from omeify.converters import Raw2OmeTiffConverter
from omeify.utils.generate_ome_xml import generate_ome_xml

import xml.dom.minidom


from datetime import datetime

import os
import logging
import hashlib
import time

class GenericConversion:
    def __init__(self, input_file_path, series = 0, rename_channels = {}):
        self.input_file_path = input_file_path
        self._rename_channels = rename_channels
        self._series = series
        self._cache_directory = None
        self.logger = logging.getLogger(__name__)

    @property
    def cache_directory(self):
        return self._cache_directory
    @cache_directory.setter
    def cache_directory(self, value):
        self._cache_directory = value

    @property
    def series(self):
        return self._series
    @series.setter
    def series(self, value):
        self._series = value
    
    @property
    def rename_channels(self):
        return self._rename_channels    
    @rename_channels.setter
    def rename_channels(self, value):
        self._rename_channels = value

    @property
    def input_type(self):
        # String representation of the input type
        raise NotImplementedError("Needs to be implemented for the specific input type.")

    def raw2ometiff(self,zarr,output_path):
        #Raw2OmeTiffConverter(zarr.store.path).convert(output_path)
        raise NotImplementedError("Needs to be implemented for the specific input type.")

    def generate_original_tiff_features(self):
        # Implement the generic OME-TIFF metadata update step here
        # Will use self.input_file_path and self.series to generate ome tiff features
        # This may involve using the specific ImageFeatures class to parse the input
        raise NotImplementedError("Needs to be implemented for the specific input type.")

    def convert(self, output_path, display_uuid = True, deidentify_ome = True, compression = 'LZW'):
        start_time = time.time()

        from omeify.utils import OMESchemaValidator
        from omeschema import get_ome_schema_path
        from omeify import get_version_info

        # Convert the currently selected series
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Converting Series [{self.series}] into OME-TIFF")
        b2r_converter = Bioformats2RawConverter(self.input_file_path)
        zarr = b2r_converter.convert(series = self.series, cache_directory = self.cache_directory)
        tf = self.generate_original_tiff_features()
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Constructing OME metadata...")
        _d = generate_ome_xml(tf, zarr, display_uuid = display_uuid, rename_channels = self.rename_channels)
        omexml = _d['xml_string']
        myuuid = _d['uuid']

        # Parse the XML string
        dom = xml.dom.minidom.parseString(omexml)

        # Pretty-print the XML
        prettyxml = dom.toprettyxml(indent="  ")



        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Constructed OME metadata:\n{prettyxml}")
        if deidentify_ome:
            # now replace the METADATA.ome.xml
            with open(os.path.join(zarr.store.path,'OME','METADATA.ome.xml'),'w') as output_file:
                output_file.write(omexml)
        else:
            with open(os.path.join(zarr.store.path,'OME','METADATA.ome.xml'),'rt') as inf:
                omexml = inf.read()
        if os.path.exists(output_path):
            self.logger.info("Overwriting existing output file by removing it first.")
            os.remove(output_path)
        self.raw2ometiff(zarr,output_path,compression)
        b2r_converter.cleanup()
        osv = OMESchemaValidator(schema_location = get_ome_schema_path())

        # now get file information
        input_size = os.path.getsize(self.input_file_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = output_size / input_size


        stop_time = time.time()

        run_time = stop_time - start_time

        # make run times readable
        start_time_readable = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        stop_time_readable = datetime.fromtimestamp(stop_time).strftime('%Y-%m-%d %H:%M:%S')
        run_time_readable = f"{run_time // 3600:02.0f}:{(run_time % 3600) // 60:02.0f}:{run_time % 60:05.2f}"

        return {
            'ome':{
                'xml_string':omexml,
                'schema_location':osv.schema_location,
                'xml_is_valid':osv.validate(omexml),
                'uuid':myuuid,
            },
            'input_file':{
                'path':self.input_file_path,
                'md5_checksum': md5_checksum(self.input_file_path),
                'size_bytes':input_size,
                'type_description':self.input_type,
            },
            'output_file':{
                'path':output_path,
                'md5_checksum': md5_checksum(output_path),
                'size_bytes':output_size,
                'type_description':'OME-TIFF',
            },
            'options':{
                'compression':compression,
                'deidentify_ome':deidentify_ome,
                'display_uuid':display_uuid,
                'series':self.series,
                'rename_channels':self.rename_channels,
            },
            'conversion_stats':{
                'start_time': start_time_readable,
                'stop_time': stop_time_readable,
                'run_time': run_time_readable,
                'compression_ratio':compression_ratio,
            },
            'versions':get_version_info()
        }

def md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
