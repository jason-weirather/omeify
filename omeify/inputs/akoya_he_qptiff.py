from omeify.utils import GenericConversion
#from omeify.inputs.image_features import AkoyaHEQptiffImageFeatures
from omeify.utils.tiff_image_features import TiffImageFeatures
from omeify.converters import Raw2OmeTiffConverter


class AkoyaHEQptiff(GenericConversion):
    def __init__(self, input_file_path, series=0):
        super().__init__(input_file_path, series=series)

    def generate_original_tiff_features(self):
        # Add code here to update the OME-TIFF metadata based on AkoyaHeTiffImageFeatures
        return AkoyaHEQptiffImageFeatures(self.input_file_path, series = self.series)

    def raw2ometiff(self,zarr,output_path):
        Raw2OmeTiffConverter(zarr.store.path).convert(output_path, rgb = True)

class AkoyaHEQptiffImageFeatures(TiffImageFeatures):
    def __init__(self, tiff_file_path, series=0):
        super().__init__(tiff_file_path, series)

    # Override the name property
    @property
    def name(self):
        # Implement the platform-specific logic for retrieving the name here
        return 'H&E'
    
    # Override the size_c property
    @property
    def size_c(self):
        # Implement the platform-specific logic for retrieving the size_c here
        return 3
    
    # Override the channels property
    @property
    def channels(self):
        channel_features = [
            {
                'ID':'Channel:0:0',
                'Name':'RGB',
                'SamplesPerPixel':3
            },
        ]
        return channel_features