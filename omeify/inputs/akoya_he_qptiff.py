from omeify.utils import GenericConversion
#from omeify.inputs.image_features import AkoyaHEQptiffImageFeatures
from omeify.utils.tiff_image_features import TiffImageFeatures
from omeify.converters import Raw2OmeTiffConverter


class AkoyaHEQptiff(GenericConversion):
    def __init__(self, input_file_path, series=0, rename_channels = {}):
        super().__init__(input_file_path, series=series, rename_channels = rename_channels)

    def generate_original_tiff_features(self):
        # Add code here to update the OME-TIFF metadata based on AkoyaHeTiffImageFeatures
        return AkoyaHEQptiffImageFeatures(self.input_file_path, series = self.series)

    def raw2ometiff(self, zarr, output_path, compression):
        Raw2OmeTiffConverter(zarr.store.path).convert(output_path, rgb = False, compression = compression)

    @property
    def input_type(self):
        # String representation of the input type
        return "Akoya HE QPTIFF"

class AkoyaHEQptiffImageFeatures(TiffImageFeatures):
    def __init__(self, tiff_file_path, series=0):
        super().__init__(tiff_file_path, series)

    # Override the name property
    @property
    def name(self):
        # Implement the platform-specific logic for retrieving the name here
        return 'WholeSlideHE'
    
    # Override the size_c property
    @property
    def size_c(self):
        # Implement the platform-specific logic for retrieving the size_c here
        return 3

    @property
    def plane_count(self):
        return 3
        
    # Override the channels property
    @property
    def channels(self):
        '''
        channel_features = [
            {
                'ID':'Channel:00',
                'Name':'RGB',
                'SamplesPerPixel':3
            },
        ]
        '''
        channel_features = [
            {
                'ID':'Channel:0:0',
                'Name':'RED',
                'SamplesPerPixel':1
            },
            {
                'ID':'Channel:0:1',
                'Name':'GREEN',
                'SamplesPerPixel':1
            },
            {
                'ID':'Channel:0:2',
                'Name':'BLUE',
                'SamplesPerPixel':1
            },
        ]
        return channel_features