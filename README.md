# omeify

`omeify` is a Python package designed to streamline the conversion of TIFF files into the OME-TIFF format, which has been deidentified using the MITI standard. This package relies on two Java tools: [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) and [raw2ometiff](https://github.com/glencoesoftware/raw2ometiff).

## Dependencies

- `bioformats2raw`: A Java application that converts various image file formats, including .mrxs, to an intermediate Zarr structure compatible with the OME-NGFF specification. This tool is used in conjunction with `raw2ometiff` to produce a Bio-Formats 5.9.x ("Faas") or Bio-Formats 6.x (true OME-TIFF) pyramid.
- `raw2ometiff`: A Java application that converts a directory of tiles to an OME-TIFF pyramid. This tool is the second half of the iSyntax/.mrxs to OME-TIFF conversion process.

**Note**: As `omeify` is licensed under the MIT license, the GPL-licensed dependencies (`bioformats2raw` and `raw2ometiff`) are not included. Instructions on how to install these dependencies will be provided later.

## MITI Standard

`omeify` follows the [Minimum Information guidelines for highly multiplexed tissue images (MITI)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9009186/) to ensure the highest standards in data and metadata handling. The MITI standard is specifically designed for tissue atlases that combine multi-channel microscopy with single cell sequencing and other omics data from normal and diseased specimens. This standard guides data deposition, curation, and release.

## Installation

*Instructions on how to install `omeify` and its dependencies will be added here.*

## Usage

*Instructions on how to use `omeify` will be added here.*

## License

This project is licensed under the MIT License. Please note that the `bioformats2raw` and `raw2ometiff` dependencies are licensed under the GPL License and are not included in this repository.

