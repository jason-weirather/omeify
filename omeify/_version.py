from omeify.converters.bioformats2raw_converter import Bioformats2RawConverter
from omeify.converters.raw2ometiff_converter import Raw2OmeTiffConverter
from tiffinspector import __version__ as my_tiffinspector_version

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version
__version__ = version('omeify')

def get_version_info():

    version_info = {
        'omeify': __version__,
        'bioformats2raw': Bioformats2RawConverter.get_version(),
        'raw2ometiff': Raw2OmeTiffConverter.get_version(),
        'tiff-inspector': my_tiffinspector_version,
    }
    return version_info

