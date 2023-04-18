# cli.py
import argparse
import logging
from .inputs import AkoyaMIFQptiff, AkoyaHEQptiff

def main():
    parser = argparse.ArgumentParser(description='omeify: Convert images into OME-TIFF format', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output OME-TIFF file path')
    parser.add_argument('--type', choices=['qptiff_mif', 'qptiff_he'], 
                                  required=True, 
                                  help='Input image type (qptiff_mif: Akoya mIF qptiff, qptiff_he: Akoya H&E qptiff)')
    parser.add_argument('--series', type=int, default=0, help='Series number (integer)')
    parser.add_argument('--omit_uuid', action='store_true', help='Omit UUID in OME tag')
    parser.add_argument('--uuid_output', type=str, help='Output file for UUID')
    parser.add_argument('-v','--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)

    if args.type == 'qptiff_mif':
        input_processor = AkoyaMIFQptiff(args.input,series=args.series)
    elif args.type == 'qptiff_he':
        input_processor = AkoyaHEQptiff(args.input,series=args.series)

    _d = input_processor.convert(args.output,display_uuid = False if args.omit_uuid else True)
    myuuid = _d['uuid']

    if args.uuid_output and not args.omit_uuid:
        with open(args.uuid_output, 'w') as f:
            f.write(myuuid)
    elif not args.omit_uuid:
        print(myuuid)

if __name__ == '__main__':
    main()
