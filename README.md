# omeify

`omeify` is a Python package designed to streamline the conversion of various image files, such as TIFF files, into the OME-TIFF format. The generated OME-TIFF files are deidentified following the MITI standard. This package relies on two Java tools: [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) `0.6.1` and [raw2ometiff](https://github.com/glencoesoftware/raw2ometiff) `0.4.1`.

## Docker quickstart

```
$ docker pull vacation/omeify:latest
$ docker run --rm -v $(pwd):$(pwd) -u $(id -u):$(id -g) vacation/omeify:latest omeify -h
usage: omeify [-h] --type {qptiff_mif,qptiff_he} [--series SERIES]
              [--rename_channels_json RENAME_CHANNELS_JSON] [--omit_uuid]
              [--output_json OUTPUT_JSON] [--cache_directory CACHE_DIRECTORY]
              [--compression {LZW,JPEG,Uncompressed}] [-v] [--version]
              input output

omeify: Convert images into OME-TIFF format

positional arguments:
  input                 Input image file path
  output                Output OME-TIFF file path

options:
  -h, --help            show this help message and exit
  --type {qptiff_mif,qptiff_he}
                        Input image type (qptiff_mif: Akoya mIF qptiff,
                        qptiff_he: Akoya H&E qptiff) (default: None)
  --series SERIES       Series number (integer) (default: 0)
  --rename_channels_json RENAME_CHANNELS_JSON
                        JSON file that contains channel renaming dictionary
                        (default: None)
  --omit_uuid           Omit UUID in OME tag (default: False)
  --output_json OUTPUT_JSON
                        Output file for run info (default: None)
  --cache_directory CACHE_DIRECTORY
                        Path to a directory for storing temporary Zarr
                        directories. Defaults to the system temporary folder.
                        (default: None)
  --compression {LZW,JPEG,Uncompressed}
                        Compression type for output OME-TIFF file (LZW, JPEG)
                        (default: LZW)
  -v, --verbose         Enable verbose logging (default: False)
  --version             Display omeify and constituent programs versions
                        (default: False)
```

## Dependencies

- `bioformats2raw`: A Java application that converts various image file formats, including .mrxs, to an intermediate Zarr structure compatible with the OME-NGFF specification. This tool is used in conjunction with `raw2ometiff` to produce a Bio-Formats 5.9.x ("Faas") or Bio-Formats 6.x (true OME-TIFF) pyramid.
- `raw2ometiff`: A Java application that converts a directory of tiles to an OME-TIFF pyramid. This tool is the second half of the iSyntax/.mrxs to OME-TIFF conversion process.

**Note**: As `omeify` is licensed under the MIT license, the GPL-licensed dependencies (`bioformats2raw` and `raw2ometiff`) are not included. Instructions on how to install these dependencies will be provided later.

## MITI Standard

`omeify` follows the [Minimum Information guidelines for highly multiplexed tissue images (MITI)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9009186/) to ensure the highest standards in data and metadata handling. The MITI standard is specifically designed for tissue atlases that combine multi-channel microscopy with single cell sequencing and other omics data from normal and diseased specimens. This standard guides data deposition, curation, and release.

## Installation

1. Install the `omeify` package:

```bash
pip install omeify
```

## Usage

### Command Line Interface (CLI)

You can use `omeify` through the command line interface by running the following command:

```
$ omeify -h
usage: omeify [-h] --type {qptiff_mif,qptiff_he} [--series SERIES]
              [--rename_channels_json RENAME_CHANNELS_JSON] [--omit_uuid] [--output_json OUTPUT_JSON]
              [--cache_directory CACHE_DIRECTORY] [--compression {LZW,JPEG,Uncompressed}] [-v]
              [--version]
              input output

omeify: Convert images into OME-TIFF format

positional arguments:
  input                 Input image file path
  output                Output OME-TIFF file path

options:
  -h, --help            show this help message and exit
  --type {qptiff_mif,qptiff_he}
                        Input image type (qptiff_mif: Akoya mIF qptiff, qptiff_he: Akoya H&E qptiff)
                        (default: None)
  --series SERIES       Series number (integer) (default: 0)
  --rename_channels_json RENAME_CHANNELS_JSON
                        JSON file that contains channel renaming dictionary (default: None)
  --omit_uuid           Omit UUID in OME tag (default: False)
  --output_json OUTPUT_JSON
                        Output file for run info (default: None)
  --cache_directory CACHE_DIRECTORY
                        Path to a directory for storing temporary Zarr directories. Defaults to the
                        system temporary folder. (default: None)
  --compression {LZW,JPEG,Uncompressed}
                        Compression type for output OME-TIFF file (LZW, JPEG) (default: LZW)
  -v, --verbose         Enable verbose logging (default: False)
  --version             Display omeify and constituent programs versions (default: False)
```

## Python API

You can also use `omeify` within your Python scripts:

```py
from omeify.inputs import AkoyaMIFQptiff, AkoyaHEQptiff

input_processor = AkoyaMIFQptiff(input_file_path, series=series_number)
input_processor.rename_channels = rename_channels_dict

output_info = input_processor.convert(output_file_path, display_uuid=True)
```

Replace `AkoyaMIFQptiff` with `AkoyaHEQptiff` if you are working with H&E qptiff files.

## License

This project is licensed under the MIT License. Please note that the `bioformats2raw` and `raw2ometiff` dependencies are licensed under the GPL License and are not included in this repository.

