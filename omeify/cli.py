# cli.py
import argparse, json, logging, sys
from .inputs import AkoyaMIFQptiff, AkoyaHEQptiff, AkoyaComponentTiff

def main():
    if '--version' in sys.argv:
        from omeify import get_version_info
        print(json.dumps(get_version_info(), indent=2))
        sys.exit(0)

    parser = argparse.ArgumentParser(description='omeify: Convert images into OME-TIFF format', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output OME-TIFF file path')
    parser.add_argument('--type', choices=['qptiff_mif', 'qptiff_he','component'], 
                                  required=True, 
                                  help='Input image type (qptiff_mif: Akoya mIF qptiff, qptiff_he: Akoya H&E qptiff, component: Akoya Component tiff)')
    parser.add_argument('--series', type=int, default=0, help='Series number (integer)')
    parser.add_argument('--rename_channels_json', type=str, help='JSON file that contains channel renaming dictionary')
    parser.add_argument('--omit_uuid', action='store_true', help='Omit UUID in OME tag')
    parser.add_argument('--output_json', type=str, help='Output file for run info')
    parser.add_argument('--cache_directory', type=str, help="Path to a directory for storing temporary Zarr directories. Defaults to the system temporary folder.")
    parser.add_argument('--compression', type=str, default='LZW', choices=['LZW', 'JPEG', 'Uncompressed'], help='Compression type for output OME-TIFF file (LZW, JPEG)')
    parser.add_argument('-v','--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='store_true', help='Display omeify and constituent programs versions')
    parser.add_argument('--physical_size_x_um', help='Provide a size in um for x pixel size for types where one is not provided')
    parser.add_argument('--physical_size_y_um', help='Provide a size in um for y pixel size for types where one is not provided')

    args = parser.parse_args()

    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)

    if args.type == 'qptiff_mif':
        input_processor = AkoyaMIFQptiff(args.input,series=args.series)
    elif args.type == 'qptiff_he':
        input_processor = AkoyaHEQptiff(args.input,series=args.series)
    elif args.type == 'component':
        if args.physical_size_x_um is None or args.physical_size_y_um is None:
            raise ValueError("When generating a component OME tiff the arguments physical_size_x_um and physical_size_y_um are required.")
        input_processor = AkoyaComponentTiff(
            args.input,
            series=args.series,
            physical_size_x_um=args.physical_size_x_um,
            physical_size_y_um=args.physical_size_y_um
        )

    if args.cache_directory:
        input_processor.cache_directory = args.cache_directory

    if args.rename_channels_json:
        _d = {}
        with open(args.rename_channels_json,'rt') as inf:
            _d = json.loads(inf.read())
        input_processor.rename_channels = _d


    output_info = input_processor.convert(args.output,display_uuid = False if args.omit_uuid else True, compression = args.compression)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(json.dumps(output_info,indent=2))
    else:
        print(json.dumps(output_info,indent=2))
    


if __name__ == '__main__':
    main()
