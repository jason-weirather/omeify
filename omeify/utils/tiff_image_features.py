import tifffile
import xmltodict
from tiffinspector import TiffInspector

class TiffImageFeatures:
    def __init__(self, tiff_file_path, series = 0):
        self.tiff_file_path = tiff_file_path
        self.report = TiffInspector(tiff_file_path).report
        self._image_description = None
        self._series = None
        self._tags = None
        self._metadata = None

        # Subsequent properties are updated upon series selection
        self.series = series

    @property
    def metadata(self):
        # Return a dictionary of the metadata first page of the first level of the currently selected series.
        return self._metadata
 
    @property
    def tags(self):
        # Return a dictionary of the tags of the first page of the first level of the currently selected series.
        return self._tags
 
    @property
    def image_description(self):
        # Return the image description of the first page of the first level of the currently selected series.
        return self._image_description
 

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        # Upon updating the series, update the tags and image description
        self._series = value
        self._metadata = self.report['series'][value]['levels'][0]['pages'][0]['metadata']
        _tags0 = self.report['series'][value]['levels'][0]['pages'][0]['tags']
        self._tags = dict([(x[0],x[4]) for x in _tags0])
        self._image_description = xmltodict.parse(self._tags['ImageDescription'])

    # Image properties
    @property
    def image_id(self):
        return "Image:0"

    @property
    # Needs to be overridden
    def name(self):
        raise NotImplementedError("Needs to be overridden by the extended class.")

    # Pixels properties
    @property
    def big_endian(self):
        # Set according to conversion scripts
        return "true"
        #return False if self.report['metadata']['byteorder'] == '<' else True

    @property
    def dimension_order(self):
        # Set according to conversion scripts
        return "XYZCT"

    @property
    def pixel_id(self):
        # Set according to conversion scripts
        return 'Pixels:0'

    @property
    def interleaved(self):
        # Set according to conversion scripts
        return "false"

    @property
    def physical_size_x(self):
        return self.image_description['PerkinElmer-QPI-ImageDescription']\
            ['ScanProfile']['root']['ScanResolution']['PixelSizeMicrons']

    @property
    def physical_size_x_unit(self):
        # Set according to conversion scripts
        return "µm"

    @property
    def physical_size_y(self):
        return self.image_description['PerkinElmer-QPI-ImageDescription']\
            ['ScanProfile']['root']['ScanResolution']['PixelSizeMicrons']

    @property
    def physical_size_y_unit(self):
        # Set according to conversion scripts
        return "µm"

    @property
    def significant_bits(self):
        # Set according to conversion scripts
        return 8

    @property
    def size_t(self):
        return self.report['series'][self.series]\
            ['levels'][0]['pages'][0]['metadata']['shaped'][0]
    
    @property
    def size_z(self):
        return self.report['series'][self.series]\
            ['levels'][0]['pages'][0]['metadata']['shaped'][1]
    
    @property
    def size_y(self):
        return self.report['series'][self.series]\
            ['levels'][0]['pages'][0]['metadata']['shaped'][2]
    
    @property
    def size_x(self):
        return self.report['series'][self.series]\
            ['levels'][0]['pages'][0]['metadata']['shaped'][3]

    @property
    def size_c(self):
        # Needs to be overriden
        raise NotImplementedError("Needs to be overriden by specific image type.")

    @property
    def type(self):
        return self.report['series'][self.series]\
            ['levels'][0]['pages'][0]['metadata']['dtype']

    @property
    def channels(self):
        # Needs to be overriden
        raise NotImplementedError("Needs to be overriden by specific image type.")
    
