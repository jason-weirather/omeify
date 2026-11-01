"""Metadata-only inspection and explicit OME ROI preservation regressions."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from lxml import etree

from omeify import TiffInspector, convert, mutate
from omeify.cli import main
from omeify.io.pixel_size import PixelSize
from omeify.io.spec import OMEImageSpec
from omeify.roi_metadata import load_preserved_rois, verify_preserved_rois
from omeify.utils.generate_ome_xml import generate_ome_xml

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _xml(*, t=1, roi_count=2, ref=True, the_t=None, extra_annotation=False):
    refs = ''.join(f'<ROIRef ID="ROI:{i}"/>' for i in range(roi_count)) if ref else ''
    rois = []
    for i in range(roi_count):
        plane = '' if the_t is None else f' TheT="{the_t}"'
        annotation = '<AnnotationRef ID="Annotation:0"/>' if extra_annotation else ''
        rois.append(
            f'<ROI ID="ROI:{i}" Name="ROI {i}">'
            f'<Union><Rectangle ID="Shape:{i}" X="1" Y="2" Width="3" '
            f'Height="4"{plane}/></Union>{annotation}</ROI>'
        )
    return (
        f'<?xml version="1.0" encoding="utf-8"?>'
        f'<OME xmlns="{NS}"><Image ID="Image:0">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="float" '
        f'SizeX="16" SizeY="16" SizeZ="1" SizeC="1" SizeT="{t}" '
        f'PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
        '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
        f'<TiffData IFD="0" PlaneCount="{t}"/>'
        f'</Pixels>{refs}</Image>{"".join(rois)}</OME>'
    )


def _input(path, *, t=1, roi_count=2, ref=True, the_t=None, extra_annotation=False):
    array = np.arange(t * 16 * 16, dtype=np.float32).reshape(t, 16, 16) / 10
    if t == 1:
        array = array[0]
    tifffile.imwrite(
        path, array,
        description=_xml(t=t, roi_count=roi_count, ref=ref, the_t=the_t,
                         extra_annotation=extra_annotation).encode('utf-8'),
        photometric='minisblack', metadata=None,
    )
    return array


def _spec(t=1):
    axes = 'YX' if t == 1 else 'TCZYX'
    shape = (16, 16) if t == 1 else (t, 1, 1, 16, 16)
    return OMEImageSpec('multichannel', axes, shape, np.dtype('float32'),
                        ('Signal',), PixelSize(.5, .5, 'µm'))


def test_inspect_counts_rois_without_expanding_thousands_of_shapes(tmp_path):
    path = tmp_path / 'many.ome.tiff'
    _input(path, roi_count=3000)
    inspector = TiffInspector(path, detail=0)
    report = inspector.report
    assert report['ome']['rois'] == {'count': 3000, 'shapes': 3000, 'image_references': 3000}
    assert len(report['ome']['miti']['extra_metadata']) <= 65
    assert any('ROI: 3000 element(s)' in value
               for value in report['ome']['miti']['extra_metadata'])
    assert inspector.validation_errors() == ()
    tree = inspector.render_text()
    assert '3000 ROI(s), 3000 shape(s), 3000 image reference(s)' in tree
    assert 'ROI:2999' not in tree
    assert len(tree) < 15_000


def test_default_generated_ome_xml_drops_source_roi_annotations(tmp_path):
    path = tmp_path / 'source.ome.tiff'
    _input(path)
    rois = load_preserved_rois(path)
    assert rois.count == 2
    minimal = generate_ome_xml(_spec(), ((16, 16),))['xml_string']
    root = etree.fromstring(minimal.encode())
    assert not root.findall(f'{{{NS}}}ROI')
    assert not root.findall(f'.//{{{NS}}}ROIRef')


def test_preserve_linked_rois_roundtrip_without_decoding_masks(tmp_path):
    source = tmp_path / 'source.ome.tiff'
    target = tmp_path / 'rebuilt.ome.tiff'
    image = _input(source)
    rois = load_preserved_rois(source)
    xml = generate_ome_xml(_spec(), ((16, 16),), preserved_rois=rois)['xml_string']
    # tifffile requires encoded bytes for non-ASCII OME units (µm).
    tifffile.imwrite(target, image, description=xml.encode('utf-8'), metadata=None)
    verify_preserved_rois(target, rois)
    assert TiffInspector(target, detail=0).report['ome']['rois']['count'] == 2


def test_preservation_rejects_changed_time_and_unresolved_references(tmp_path):
    path = tmp_path / 'movie.ome.tiff'
    _input(path, t=3, the_t=2)
    rois = load_preserved_rois(path)
    assert rois.size_t == 3
    with pytest.raises(ValueError, match='unchanged X/Y/Z/T/C'):
        generate_ome_xml(_spec(), ((16, 16),), preserved_rois=rois)
    original = generate_ome_xml(_spec(t=3), ((3, 1, 1, 16, 16),), preserved_rois=rois)
    root = etree.fromstring(original['xml_string'].encode())
    assert root.find(f'.//{{{NS}}}Rectangle').get('TheT') == '2'

    unlinked = tmp_path / 'orphan.ome.tiff'
    _input(unlinked, ref=False)
    with pytest.raises(ValueError, match='unassociated ROIs'):
        load_preserved_rois(unlinked)


def test_preservation_rejects_unretained_annotation_dependencies(tmp_path):
    source = tmp_path / 'annotated.ome.tiff'
    _input(source, extra_annotation=True)
    with pytest.raises(ValueError, match='external annotations'):
        load_preserved_rois(source)


def test_cli_exposes_explicit_flag_and_rejects_unsupported_formats():
    cli = CliRunner()
    for operation in ('convert', 'mutate'):
        help_result = cli.invoke(main, [operation, '--help'])
        assert help_result.exit_code == 0
        assert '--preserve-rois' in help_result.output
    with pytest.raises(ValueError, match='OME-TIFF input and OME-TIFF output'):
        convert('missing.zarr', 'output.tif', input_type='ome_zarr', preserve_rois=True)
    with pytest.raises(ValueError, match='OME-TIFF input and OME-TIFF output'):
        mutate('missing.ome.tif', 'output.zarr', input_type='ome_tiff',
               output_format='ome_zarr', float32_mantissa_bits=11, preserve_rois=True)


def test_real_convert_and_mutate_default_drop_opt_in_keeps_rois(tmp_path):
    """Uses real ome-schema validation, TIFF encoders, and readback in release test lanes."""
    source = tmp_path / 'source.ome.tiff'
    data = _input(source)
    for operation in ('convert', 'mutate'):
        for preserve in (False, True):
            dest = tmp_path / f'{operation}-{preserve}.ome.tiff'
            kwargs = dict(input_type='ome_tiff', compression='Uncompressed',
                          tile_size=16, pyramid_levels=0, preserve_rois=preserve)
            if operation == 'mutate':
                mutate(source, dest, float32_mantissa_bits=11, **kwargs)
            else:
                convert(source, dest, **kwargs)
            with tifffile.TiffFile(dest) as tiff:
                assert bool(tiff.ome_metadata)
                actual = tiff.series[0].asarray()
                assert actual.dtype == np.dtype('float32')
                assert actual.shape == data.shape
                if operation == 'convert':
                    np.testing.assert_array_equal(actual, data)
                root = etree.fromstring(tiff.ome_metadata.encode())
                assert len(root.findall(f'{{{NS}}}ROI')) == (2 if preserve else 0)
                assert len(root.findall(f'.//{{{NS}}}ROIRef')) == (2 if preserve else 0)


def test_real_stack_convert_and_mutate_keep_roi_plane_coordinates(tmp_path):
    source = tmp_path / 'source.ome.tiff'
    _input(source, t=3, the_t=2)
    for operation in ('convert', 'mutate'):
        dest = tmp_path / f'{operation}.ome.tiff'
        kwargs = dict(input_type='ome_tiff', compression='Uncompressed',
                      tile_size=16, pyramid_levels=0, preserve_rois=True)
        if operation == 'convert':
            convert(source, dest, **kwargs)
        else:
            mutate(source, dest, float32_mantissa_bits=11, **kwargs)
        with tifffile.TiffFile(dest) as tiff:
            root = etree.fromstring(tiff.ome_metadata.encode())
            assert len(root.findall(f'{{{NS}}}ROI')) == 2
            assert {n.get('TheT') for n in root.findall(f'.//{{{NS}}}Rectangle')} == {'2'}
        with pytest.raises(ValueError, match='unchanged X/Y/Z/T/C'):
            fn = convert if operation == 'convert' else mutate
            extra = {} if operation == 'convert' else {'float32_mantissa_bits': 11}
            fn(source, tmp_path / f'{operation}-subset.ome.tif', t=0, **kwargs, **extra)


def test_real_cli_convert_and_mutate_opt_in(tmp_path):
    source = tmp_path / 'cli.ome.tif'
    _input(source)
    cli = CliRunner()
    for operation in ('convert', 'mutate'):
        dest = tmp_path / f'cli-{operation}.ome.tif'
        args = [operation, str(source), '-o', str(dest), '--type', 'ome_tiff',
                '--compression', 'Uncompressed', '--tile-size', '16',
                '--pyramid-levels', '0', '--preserve-rois']
        if operation == 'mutate':
            args += ['--float32-mantissa-bits', '11']
        result = cli.invoke(main, args)
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert TiffInspector(dest, detail=0).report['ome']['rois']['count'] == 2
