from omeify.utils import GenericConversion
#from omeify.inputs.image_features import AkoyaHEQptiffImageFeatures
from omeify.utils.tiff_image_features import TiffImageFeatures
from omeify.converters import Raw2OmeTiffConverter

import xmltodict


class HaloMIFTiff(GenericConversion):
    def __init__(self, input_file_path, series=0, rename_channels = {}):
        super().__init__(input_file_path, series=series, rename_channels = rename_channels)

    def generate_original_tiff_features(self):
        # Add code here to update the OME-TIFF metadata based on AkoyaHeTiffImageFeatures
        return HaloMIFTiffImageFeatures(self.input_file_path, series = self.series)

    def raw2ometiff(self, zarr, output_path, compression):
        Raw2OmeTiffConverter(zarr.store.path).convert(output_path, rgb = False, compression = compression)

    @property
    def input_type(self):
        # String representation of the input type
        return "Halo MIF TIFF"

class HaloMIFTiffImageFeatures(TiffImageFeatures):
    def __init__(self, tiff_file_path, series=0):
        super().__init__(tiff_file_path, series)

    # Override the name property
    @property
    def name(self):
        # Implement the platform-specific logic for retrieving the name here
        return 'WholeSlideMIF'
    
    # Override the physical_size_x property
    @property
    def physical_size_x(self):
        return self.image_description['OME']\
            ['Image']['Pixels']['@PhysicalSizeX']
    
    # Override the physical_size_y property
    @property
    def physical_size_y(self):
        return self.image_description['OME']\
            ['Image']['Pixels']['@PhysicalSizeY']
    
    # Override the size_x property
    @property
    def size_x(self):
        return self.image_description['OME']\
            ['Image']['Pixels']['@SizeX']
    
    # Override the size_y property
    @property
    def size_y(self):
        return self.image_description['OME']\
            ['Image']['Pixels']['@SizeY']

    # Override the size_c property
    @property
    def size_c(self):
        # Implement the platform-specific logic for retrieving the size_c here
        #return len(self.report['series'][self.series]['levels'][0]['pages'])
        return self.image_description['OME']\
            ['Image']['Pixels']['@SizeC']

    @property
    def plane_count(self):
        #return len(self.report['series'][self.series]['levels'][0]['pages'])
        return self.image_description['OME']\
            ['Image']['Pixels']['@SizeZ']

    # Override the channels property
    @property
    def channels(self):
        channel_features = []
        # only take info from first page... page[0]
        page = self.report['series'][self.series]['levels'][0]['pages'][0]
        _tags_dict = dict([(x[0],x[4]) for x in page['tags']])
        _d = xmltodict.parse(_tags_dict['ImageDescription'])
        channel_features = _d['OME']['Image']['Pixels']['Channel']
        # replace keys with ones without @ symbols
        replacement = {'@ID':'ID',
              '@Name':'Name',
              '@SamplesPerPixel':'SamplesPerPixel',
              '@Color':'Color'}
        for d in channel_features:
            for k, v in list(d.items()):
                d[replacement.get(k, k)] = d.pop(k)
        
        return channel_features
