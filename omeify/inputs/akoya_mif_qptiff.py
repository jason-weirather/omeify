from omeify.utils import GenericConversion
#from omeify.inputs.image_features import AkoyaHEQptiffImageFeatures
from omeify.utils.tiff_image_features import TiffImageFeatures
from omeify.converters import Raw2OmeTiffConverter

import xmltodict


class AkoyaMIFQptiff(GenericConversion):
    def __init__(self, input_file_path, series=0):
        super().__init__(input_file_path, series=series)

    def generate_original_tiff_features(self):
        # Add code here to update the OME-TIFF metadata based on AkoyaHeTiffImageFeatures
        return AkoyaMIFQptiffImageFeatures(self.input_file_path, series = self.series)

    def raw2ometiff(self,zarr,output_path):
        Raw2OmeTiffConverter(zarr.store.path).convert(output_path, rgb = False)

class AkoyaMIFQptiffImageFeatures(TiffImageFeatures):
    def __init__(self, tiff_file_path, series=0):
        super().__init__(tiff_file_path, series)

    # Override the name property
    @property
    def name(self):
        # Implement the platform-specific logic for retrieving the name here
        return 'WholeSlideMIF'
    
    # Override the size_c property
    @property
    def size_c(self):
        # Implement the platform-specific logic for retrieving the size_c here
        return len(self.report['series'][self.series]['levels'][0]['pages'])
        
    # Override the channels property
    @property
    def channels(self):
        channel_features = []
        for i,page in enumerate(self.report['series'][self.series]['levels'][0]['pages']):
            _tags_dict = dict([(x[0],x[4]) for x in page['tags']])
            _d = xmltodict.parse(_tags_dict['ImageDescription'])
            c = {
                'ID':f'Channel:0:{i}',
                'Name':_d['PerkinElmer-QPI-ImageDescription']['Name'],
                'SamplesPerPixel':1
            }
            channel_features.append(c)
        return channel_features