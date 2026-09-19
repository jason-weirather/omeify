"""Storage-independent source contract, lifetime, and actual file boundary tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import tifffile

from omeify import (
    ArraySource, Image, ImageLevel, ImageMetadata, ImageSource, LabelImage,
    MultichannelImage, OMEImageSeries, OMEMultiSeriesWriter, OMETiffLabelReader,
    OMETiffReader, OMETiffSource, OMETiffWriter, PixelSize, ReaderImageSource,
    RGBImage, TemporaryOMETiffWriter, write_ometiff,
)
from omeify.io.image_planes import ImagePlaneReader, ImagePlaneSource
from omeify.utils.ome_schema_validator import OMESchemaValidator

SIZE = PixelSize(0.5, 0.6, "µm")
HAS_SCHEMA = OMESchemaValidator().schema_lxml is not None
requires_schema = pytest.mark.skipif(not HAS_SCHEMA, reason="requires installed ome-schema dependency")


class CoordinateSource(ImageSource):
    """Real on-demand evaluation: no precomputed array, path, or full-image method."""

    def __init__(self, kind="multichannel", shape=None, *, calibrated=True, broken=None):
        shapes = {"multichannel": (3, 65, 97), "rgb": (65, 97, 3), "label": (65, 97)}
        axes = {"multichannel": "CYX", "rgb": "YXS", "label": "YX"}
        super().__init__(ImageMetadata(
            axes[kind], shape or shapes[kind], np.uint8 if kind == "rgb" else np.uint32,
            image_type=kind, pixel_size=SIZE if calibrated else None,
        ))
        self.calls = []
        self.opens = 0
        self.closes = 0
        self.broken = broken

    def _open(self):
        self.opens += 1

    def _close(self):
        self.closes += 1

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        self.calls.append((y0, y1, x0, x1, level, channels))
        if self.broken == "exception":
            raise RuntimeError("intentional source failure")
        y = np.arange(y0, y1, dtype=np.uint32)[:, None]
        x = np.arange(x0, x1, dtype=np.uint32)[None, :]
        if self.metadata.image_type == "label":
            values = (y // 8) * np.uint32(1000) + x // 8 + np.uint32(1)
        else:
            values = np.stack([((y * 7 + x * 3 + c * 11) % 251) for c in channels])
            if self.metadata.image_type == "rgb":
                values = np.moveaxis(values.astype(np.uint8), 0, -1)
        if self.broken == "shape":
            return values[..., :-1]
        if self.broken == "dtype":
            return values.astype(np.float64)
        if self.broken == "list":
            return values.tolist()
        return values


def expected(kind, y0, y1, x0, x1, channels=(0, 1, 2)):
    # Independent scalar oracle, not a second call to the source.
    if kind == "label":
        return np.array([[int(y // 8) * 1000 + int(x // 8) + 1 for x in range(x0, x1)]
                         for y in range(y0, y1)], dtype=np.uint32)
    planes = np.array([[[int((y * 7 + x * 3 + c * 11) % 251) for x in range(x0, x1)]
                       for y in range(y0, y1)] for c in channels], dtype=np.uint32)
    return np.moveaxis(planes.astype(np.uint8), 0, -1) if kind == "rgb" else planes


@pytest.mark.parametrize("kind,wrapper", [
    ("multichannel", MultichannelImage), ("rgb", RGBImage), ("label", LabelImage),
])
def test_genuinely_virtual_huge_source_and_tiling(kind, wrapper):
    huge = {"multichannel": (3, 100_000, 120_000), "rgb": (100_000, 120_000, 3),
            "label": (100_000, 120_000)}[kind]
    source = CoordinateSource(kind, huge)
    assert not hasattr(source, "path")
    with wrapper(source) as image:
        assert image.metadata.shape == huge
        assert source.calls == []
        full = image.read_region(49_991, 50_016, 61_993, 62_012)
        np.testing.assert_array_equal(full, expected(kind, 49_991, 50_016, 61_993, 62_012))
        top = image.read_region(49_991, 50_000, 61_993, 62_012)
        bottom = image.read_region(50_000, 50_016, 61_993, 62_012)
        axis = 1 if kind == "multichannel" else 0
        np.testing.assert_array_equal(np.concatenate((top, bottom), axis=axis), full)
        np.testing.assert_array_equal(image.read_region(49_991, 50_016, 61_993, 62_012), full)
        full[...] = 0
        np.testing.assert_array_equal(image.read_region(49_991, 50_016, 61_993, 62_012),
                                      expected(kind, 49_991, 50_016, 61_993, 62_012))
    assert source.closes == 1 and image.closed
    assert all(y1-y0 <= 25 and x1-x0 <= 19 for y0,y1,x0,x1,_,_ in source.calls)


@pytest.mark.parametrize("axes,kind,shape", [
    ("CYX", "multichannel", (3, 5, 7)), ("YX", "multichannel", (5, 7)),
    ("YXS", "rgb", (5, 7, 3)), ("YX", "label", (5, 7)),
])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "int32", "float32", ">u2"])
def test_array_read_shape_dtype_and_ownership(axes, kind, shape, dtype):
    a = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
    if kind == "label" and np.dtype(dtype).kind == "f":
        with pytest.raises(ValueError, match="integer"):
            ArraySource(a, axes=axes, image_type=kind)
        return
    snapshot = a.copy()
    with Image.from_source(ArraySource(a, axes=axes, image_type=kind)) as image:
        values = image.read_region(1, 4, 2, 6)
        reference = a[:, 1:4, 2:6] if axes == "CYX" else a[1:4, 2:6, ...]
        np.testing.assert_array_equal(values, reference)
        assert values.dtype.isnative and values.flags.c_contiguous and values.flags.owndata
        assert not np.shares_memory(values, a)
        values[...] = 0
        np.testing.assert_array_equal(a, snapshot)
        if kind != "label":
            logical = image[0].read_region(1, 4, 2, 6)
            assert logical.shape == ((3, 4, 3) if kind == "rgb" else (3, 4))


def test_array_copy_and_borrow_are_explicit():
    a = np.arange(20, dtype=np.uint16).reshape(4, 5)
    with MultichannelImage.from_array(a, copy=True) as copied:
        a[:] = 0
        assert copied.asarray().max() == 19
    with MultichannelImage.from_array(a, copy=False) as borrowed:
        assert borrowed.asarray().max() == 0
    assert a.flags.writeable  # Construction never changes the caller's write flags.


def test_noncontiguous_array_and_nan_payloads():
    raw = np.array([0, 0x80000000, 0x7fc00123, 0xffc00456, 0x7f800000, 0xff800000], np.uint32)
    native = raw.view(np.float32).reshape(2, 3)
    big = native.byteswap().view(np.dtype(">f4"))[:, ::-1]
    with MultichannelImage.from_array(big) as image:
        out = image.asarray()
        np.testing.assert_array_equal(out.view(np.uint32), raw.reshape(2, 3)[:, ::-1])


def test_logical_channels_samples_and_preserved_c_axis():
    with MultichannelImage(CoordinateSource()) as image:
        np.testing.assert_array_equal(image.read_region(1, 3, 2, 5, channels=[2, 0, 2]),
                                      expected("multichannel", 1, 3, 2, 5, (2, 0, 2)))
        assert image.read_region(1, 3, 2, 5, channels=1).shape == (1, 2, 3)
        assert image[1].read_region(1, 3, 2, 5).shape == (2, 3)
    with RGBImage(CoordinateSource("rgb")) as image:
        assert image.logical_channel_count == 1 and image.sample_count == 3
        assert image.channel_names == ("RGB",)
        assert image[0].read_region(0, 2, 0, 4).shape == (2, 4, 3)
        assert image.read_region(0, 2, 0, 4, channels=[2, 0]).shape == (2, 4, 2)


@pytest.mark.parametrize("selection", [True, np.bool_(False), [True], [1.5], "0", [], 3, [-1]])
def test_invalid_channel_selection_does_not_reach_backend(selection):
    source = CoordinateSource()
    with MultichannelImage(source) as image:
        with pytest.raises((TypeError, ValueError, IndexError)):
            image.read_region(0, 3, 0, 4, channels=selection)
        assert not source.calls


@pytest.mark.parametrize("bounds", [(-1,2,0,3), (0,66,0,3), (3,2,0,3), (0,2,4,3),
                                    (0.1,2,0,3), (True,2,0,3), (0,2,0,98)])
def test_invalid_bounds_do_not_reach_backend(bounds):
    source = CoordinateSource()
    with MultichannelImage(source) as image:
        with pytest.raises((TypeError, ValueError)):
            image.read_region(*bounds)
        assert not source.calls


@pytest.mark.parametrize("kind", ["multichannel", "rgb", "label"])
def test_empty_spatial_read_does_not_render(kind):
    source = CoordinateSource(kind)
    with Image.from_source(source) as image:
        out = image.read_region(2,2,0,5)
        assert out.size == 0 and not source.calls


@pytest.mark.parametrize("level", [True, 0.1, "0", -1, 1])
def test_invalid_levels(level):
    with MultichannelImage(CoordinateSource()) as image:
        with pytest.raises((TypeError, ValueError, IndexError)):
            image.read_region(0,1,0,1,level=level)
        with pytest.raises((TypeError, ValueError, IndexError)):
            image[0].asarray(level=level)


@pytest.mark.parametrize("broken,error", [("shape",ValueError),("dtype",TypeError),
                                          ("list",TypeError),("exception",RuntimeError)])
def test_malformed_backend_never_silently_casts_or_pads(broken,error):
    source = CoordinateSource(broken=broken)
    with MultichannelImage(source) as image:
        with pytest.raises(error):
            image.read_region(0,3,0,4)
    assert source.closes == 1


def test_descriptor_validation_and_immutability():
    d = ImageMetadata("CYX", [2,5,7], np.float32, channel_names=["A","A"])
    assert d.shape == (2,5,7) and d.channel_names == ("A","A")
    with pytest.raises(FrozenInstanceError):
        d.shape = (2,6,7)
    bad = [dict(axes="YXC"), dict(shape=(2,5.1,7)), dict(shape=(2,0,7)),
           dict(channel_names=("A",)), dict(dtype=object), dict(dtype=np.complex64),
           dict(channel_ids=("same","same")), dict(channel_ids=(None,"Channel:0:0")),
           dict(icc_profile=b"bad"), dict(pixel_size=(.5,.5))]
    for kwargs in bad:
        with pytest.raises((ValueError, TypeError)):
            replace(d, **kwargs)
    with MultichannelImage.from_array(np.zeros((2,5,7)),channel_names=("A","A")) as image:
        with pytest.raises(ValueError, match="ambiguous"):
            image.get_by_name("A")
        assert image.get_by_id("Channel:0:1").index == 1


def test_multiresolution_metadata_is_explicit_even_for_odd_shapes():
    base = np.arange(35,dtype=np.uint16).reshape(5,7)
    small = base[::2,::2]
    levels = (ImageLevel(0,"YX",base.shape,SIZE),
              ImageLevel(1,"YX",small.shape,SIZE.scaled(2),(2,2)))
    with MultichannelImage.from_array(base,pixel_size=SIZE,level_arrays=(small,),levels=levels) as im:
        assert im.level_count == 2 and im.is_pyramidal
        assert im.level_shape(1) == (3,4)
        assert im.level_downsample(1) == (2,2)  # Not 5/3 or 7/4.
        np.testing.assert_array_equal(im[0].asarray(level=1),small)
    unknown = (levels[0], ImageLevel(1,"YX",small.shape))
    with MultichannelImage.from_array(base,pixel_size=SIZE,level_arrays=(small,),levels=unknown) as im:
        assert im.pixel_size_at_level(1) is None and im.level_downsample(1) is None
    with pytest.raises(ValueError,match="calibration"):
        ImageMetadata("YX",base.shape,np.uint16,pixel_size=SIZE,
                      levels=(levels[0],replace(levels[1],downsample_yx=(1.5,1.5))))


def test_lifetime_owned_borrowed_stale_channels_and_exception():
    source = CoordinateSource()
    with source:
        with MultichannelImage(source,owns_source=False) as image:
            channel = image[0]
        assert source.is_open and not image.is_open
        with pytest.raises(RuntimeError):
            channel.read_region(0,1,0,1)
        image.open()
        with pytest.raises(RuntimeError,match="earlier"):
            channel.read_region(0,1,0,1)
        image.close()
    assert source.closes == 1
    with pytest.raises(RuntimeError,match="already be open"):
        MultichannelImage(source,owns_source=False).open()
    with pytest.raises(ZeroDivisionError), MultichannelImage(source):
        raise ZeroDivisionError
    assert source.closes == 2
    source.close()
    assert source.closes == 2


def test_nested_context_rejected_without_closing_outer():
    source = CoordinateSource()
    with source:
        with pytest.raises(RuntimeError,match="Nested"), source:
            pass
        assert source.is_open
    with MultichannelImage(source) as image:
        with pytest.raises(RuntimeError,match="Nested"), image:
            pass
        assert image.is_open


def test_semantic_mismatch_does_not_leak_owned_resource():
    source = CoordinateSource("label")
    with pytest.raises(TypeError), RGBImage(source):
        pass
    assert source.closes == 1 and not source.is_open


def test_source_close_and_reopen_invalidates_borrower():
    source = CoordinateSource().open()
    with MultichannelImage(source,owns_source=False) as image:
        channel = image[0]
        source.close()
        source.open()
        with pytest.raises(RuntimeError):
            image.read_region(0,1,0,1)
        with pytest.raises(RuntimeError):
            channel.read_region(0,1,0,1)
    assert source.is_open
    source.close()


def write_fixture(path, kind, *, pyramid=False, calibrated=True):
    data = expected(kind,0,65,0,97)
    axes = {"multichannel":"CYX","rgb":"YXS","label":"YX"}[kind]
    metadata = {"axes":axes}
    if kind == "multichannel":
        metadata["Channel"] = {"Name":["DAPI","CD3","CD20"]}
    if calibrated:
        metadata.update(PhysicalSizeX=.5,PhysicalSizeY=.6,
                        PhysicalSizeXUnit="µm",PhysicalSizeYUnit="µm")
    opts = dict(photometric="rgb" if kind=="rgb" else "minisblack",tile=(16,16),compression="deflate")
    with tifffile.TiffWriter(path,ome=True) as writer:
        writer.write(data,metadata=metadata,subifds=1 if pyramid else 0,
                     resolution=(20_000,10_000/.6) if calibrated else None,
                     resolutionunit="CENTIMETER" if calibrated else None,**opts)
        if pyramid:
            reduced = data[:,::2,::2] if kind=="multichannel" else data[::2,::2,...]
            writer.write(reduced,subfiletype=1,metadata=None,
                         resolution=(10_000,5_000/.6) if calibrated else None,
                         resolutionunit="CENTIMETER" if calibrated else None,**opts)
    return data


@pytest.mark.parametrize("kind,wrapper", [("multichannel",MultichannelImage),
                                          ("rgb",RGBImage),("label",LabelImage)])
def test_actual_ome_backend_and_generic_metadata(tmp_path,kind,wrapper,monkeypatch):
    path=tmp_path/'source.ome.tif'
    data=write_fixture(path,kind,pyramid=True)
    def no_array(*args,**kwargs):
        raise AssertionError("Full raster materialization is forbidden")
    monkeypatch.setattr(tifffile.TiffPage,"asarray",no_array)
    monkeypatch.setattr(tifffile.TiffPageSeries,"asarray",no_array)
    with wrapper(OMETiffSource(path,image_type=kind)) as im:
        assert isinstance(im,wrapper)
        assert im.level_downsample(1) == pytest.approx((2,2))
        assert im.level_shape(1) == ((3,33,49) if kind=="multichannel" else (33,49,3) if kind=="rgb" else (33,49))
        assert im.backing_paths == (path,)
        np.testing.assert_array_equal(im.read_region(0,65,0,97),data)
        reference=data[:,::2,::2] if kind=="multichannel" else data[::2,::2,...]
        np.testing.assert_array_equal(im.asarray(level=1),reference)


def test_old_reader_api_and_new_borrowing_adapter(tmp_path):
    path=tmp_path/'old.ome.tif'
    write_fixture(path,"multichannel",pyramid=True)
    with OMETiffReader(path) as reader:
        assert isinstance(reader.levels[0],tifffile.TiffPageSeries)
        assert isinstance(reader.level_descriptors[0],ImageLevel)
        old_channel=reader[0]
        with reader.as_image() as image:
            np.testing.assert_array_equal(image.read_region(1,4,2,9),reader.read_region(1,4,2,9))
            assert image.channel_names == reader.channel_names
        assert reader.is_open
    reader.open()
    with pytest.raises(RuntimeError,match="earlier"):
        old_channel.read_region(0,1,0,1)
    reader.close()


def test_borrowed_reader_adapter_rejects_new_session(tmp_path):
    path=tmp_path/'old.ome.tif'
    write_fixture(path,"multichannel")
    reader=OMETiffReader(path).open()
    source=ReaderImageSource(reader)
    with MultichannelImage(source) as image:
        reader.close()
        reader.open()
        with pytest.raises(RuntimeError):
            image.read_region(0,1,0,1)
    assert reader.is_open
    reader.close()


def test_file_source_failure_cleanup_and_explicit_label_semantics(tmp_path):
    path=tmp_path/'labels.ome.tif'
    write_fixture(path,"label")
    with Image.from_source(OMETiffSource(path)) as image:
        assert isinstance(image,MultichannelImage)  # TIFF does not declare label semantics.
    bad=OMETiffSource(path,image_type="rgb")
    with pytest.raises(TypeError):
        bad.open()
    assert bad.closed and bad._reader.closed


def test_plane_adapter_is_bounded_and_stale_after_reopen():
    source=CoordinateSource()
    with MultichannelImage(source) as image:
        plane=ImagePlaneReader(image,1)
        adapter=ImagePlaneSource(image)
        assert not source.calls
        np.testing.assert_array_equal(plane.read_region(2,5,4,9),expected("multichannel",2,5,4,9,(1,))[0])
    image.open()
    with pytest.raises(RuntimeError):
        plane.read_region(0,1,0,1)
    with pytest.raises(RuntimeError):
        adapter.plane_readers()
    image.close()


@pytest.mark.parametrize("kind",["multichannel","rgb","label"])
@requires_schema
def test_writer_streams_procedural_image_and_preserves_full_pixels(tmp_path,kind,monkeypatch):
    source=CoordinateSource(kind)
    target=tmp_path/f'{kind}.ome.tif'
    with Image.from_source(source) as image:
        monkeypatch.setattr(image,"asarray",lambda **_: pytest.fail("writer used whole-image asarray"))
        report=OMETiffWriter(target,tile_size=16,compression="Deflate",overwrite=False).write(image)
        assert image.is_open
        assert report["image"]["image_type"]==kind
        assert report["pyramid"]["downsample_method"]==("nearest" if kind=="label" else "mean")
    with tifffile.TiffFile(target) as tiff:
        np.testing.assert_array_equal(tiff.asarray(),expected(kind,0,65,0,97))
        assert len(tiff.series[0].levels)>1 and tiff.is_bigtiff
    assert max(max(y1-y0,x1-x0) for y0,y1,x0,x1,_,_ in source.calls)<=32


@requires_schema
def test_multiple_sources_types_one_shared_writer(tmp_path):
    with (MultichannelImage(CoordinateSource()) as mif,
          RGBImage(CoordinateSource("rgb")) as he,
          LabelImage(CoordinateSource("label")) as labels):
        sources=(mif,he,labels)
        series=tuple(OMEImageSeries.from_image(n,im) for n,im in zip(("mIF","H&E","Labels"),sources))
        assert all(not im.source.calls for im in sources)
        report=OMEMultiSeriesWriter(tmp_path/'multi.ome.tif',tile_size=16,compression="Deflate").write(series)
        assert all(im.is_open for im in sources)
    assert report["verification"]["bigtiff"]
    with tifffile.TiffFile(tmp_path/'multi.ome.tif') as tiff:
        for i,kind in enumerate(("multichannel","rgb","label")):
            np.testing.assert_array_equal(tiff.series[i].asarray(),expected(kind,0,65,0,97))


@requires_schema
def test_writer_inference_overrides_selected_level_and_convenience(tmp_path):
    base=np.arange(35,dtype=np.float32).reshape(5,7)
    low=base[::2,::2]
    levels=(ImageLevel(0,"YX",base.shape,SIZE),ImageLevel(1,"YX",low.shape,SIZE.scaled(2),(2,2)))
    with MultichannelImage.from_array(base,pixel_size=SIZE,levels=levels,level_arrays=(low,)) as image:
        report=OMETiffWriter(tmp_path/'low.ome.tif',compression="Deflate").write(image,level=1)
        assert report["image"]["pixel_size"]==list(SIZE.scaled(2).to_tuple())
        with tifffile.TiffFile(tmp_path/'low.ome.tif') as tiff:
            np.testing.assert_array_equal(tiff.asarray(),low)
        write_ometiff(tmp_path/'convenience.ome.tif',image=image,compression="Deflate")
        with TemporaryOMETiffWriter(compression="Deflate") as writer:
            writer.write(image)
            temp=writer.path
            assert temp.exists()
        assert not temp.exists() and image.is_open


def test_writer_requires_calibration_and_does_not_reinterpret_semantics(tmp_path):
    with LabelImage(CoordinateSource("label",calibrated=False)) as image:
        with pytest.raises(ValueError,match="calibrated"):
            OMETiffWriter(tmp_path/'bad.ome.tif').write(image)
        with pytest.raises(ValueError,match="conflicts"):
            OMETiffWriter(tmp_path/'bad.ome.tif',image_type="multichannel",pixel_size=SIZE).write(image)
        assert image.is_open and not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match="explicit pixel_size"):
        OMETiffWriter(tmp_path/'array.ome.tif').write(np.zeros((3,4),np.uint8))


@pytest.mark.parametrize("alias",["same","symlink","hardlink"])
def test_writer_protects_real_file_backing_all_paths(tmp_path,alias):
    path=tmp_path/'source.ome.tif'; write_fixture(path,"multichannel")
    target=path if alias=="same" else tmp_path/'alias.ome.tif'
    if alias=="symlink": target.symlink_to(path)
    if alias=="hardlink": target.hardlink_to(path)
    original=path.read_bytes()
    with OMETiffReader(path) as reader:
        with pytest.raises(ValueError,match="backing"):
            OMETiffWriter(target).write(reader)
        with pytest.raises(ValueError,match="backing"):
            OMEMultiSeriesWriter(target).write([OMEImageSeries.from_image("same",reader)])
    assert path.read_bytes()==original


def test_array_memmap_dependency_is_protected_not_closed(tmp_path):
    path=tmp_path/'pixels.bin'
    mmap=np.memmap(path,mode='w+',dtype=np.uint16,shape=(15,19))
    mmap[:]=27
    with MultichannelImage.from_array(mmap[:,::2],pixel_size=SIZE) as image:
        with pytest.raises(ValueError,match="backing"):
            OMETiffWriter(path).write(image)
    assert mmap[0,0]==27
    mmap.flush()


@requires_schema
def test_writer_errors_preserve_destination_and_borrowed_lifetime(tmp_path):
    target=tmp_path/'existing.ome.tif'; target.write_bytes(b'unchanged')
    with MultichannelImage(CoordinateSource(broken="exception")) as image:
        with pytest.raises(RuntimeError,match="intentional"):
            OMETiffWriter(target,compression="Deflate",tile_size=16).write(image)
        assert image.is_open
    assert target.read_bytes()==b'unchanged'
    assert not list(tmp_path.glob('.omeify-*'))


@requires_schema
def test_calibration_override_does_not_mutate_source(tmp_path):
    with MultichannelImage(CoordinateSource(calibrated=False)) as image:
        report=OMETiffWriter(tmp_path/'override.ome.tif',pixel_size=SIZE,
                             channel_names=("A","B","C"),compression="Deflate").write(image)
        assert image.pixel_size is None and image.channel_names!=("A","B","C")
        assert report["image"]["channel_names"]==["A","B","C"]


def test_file_reader_exposes_session_bound_source(tmp_path):
    path=tmp_path/'reader.ome.tif'; write_fixture(path,"multichannel")
    with OMETiffReader(path) as reader:
        backend=reader.source
        assert isinstance(backend,ImageSource) and backend is reader.source
        with Image.from_source(backend,owns_source=False) as image:
            np.testing.assert_array_equal(image.read_region(1,3,2,4),reader.read_region(1,3,2,4))
        assert reader.is_open and backend.is_open
    assert backend.closed
    with pytest.raises(RuntimeError):
        backend.read_region(0,1,0,1)


def test_unknown_file_pyramid_calibration_not_inferred_from_shapes(tmp_path):
    path=tmp_path/'unknown.ome.tif'
    a=np.arange(35,dtype=np.uint16).reshape(5,7)
    with tifffile.TiffWriter(path,ome=True) as t:
        t.write(a,photometric="minisblack",subifds=1,metadata={
            "axes":"YX","PhysicalSizeX":.5,"PhysicalSizeY":.6,
            "PhysicalSizeXUnit":"µm","PhysicalSizeYUnit":"µm"})
        t.write(a[::2,::2],photometric="minisblack",subfiletype=1,metadata=None)
    with Image.from_source(OMETiffSource(path)) as image:
        assert image.level_count==2
        assert image.pixel_size_at_level(1) is None
        assert image.level_downsample(1) is None
        np.testing.assert_array_equal(image.asarray(level=1),a[::2,::2])


def test_descriptor_rejects_label_background_and_wrong_level_structure():
    with pytest.raises(ValueError):
        ImageMetadata("YX",(3,4),np.uint8,image_type="label",background_label=-1)
    with pytest.raises(ValueError):
        ImageMetadata("YX",(3,4),np.uint8,background_label=3)
    with pytest.raises(TypeError):
        ImageMetadata("CYX",(3,4,5),np.uint8,channel_names="ABC")
    with pytest.raises(ValueError):
        ImageMetadata("CYX",(3,4,5),np.uint8,levels=(
            ImageLevel(0,"CYX",(3,4,5)),ImageLevel(1,"CYX",(2,2,3))))
    with pytest.raises(ValueError):
        ImageMetadata("YX",(3,4),np.uint8,levels=(
            ImageLevel(0,"YX",(3,4)),ImageLevel(2,"YX",(2,2))))
    with pytest.raises(ValueError):
        ArraySource(np.zeros((3,4)),axes="YX",level_arrays=(np.zeros((2,2)),))


def test_rgb_custom_name_preserves_logical_identity():
    with RGBImage.from_array(np.zeros((3,4,3),np.uint8),channel_names=("Brightfield",)) as im:
        assert im.channel_names==("Brightfield",)
        assert im[0].name=="Brightfield" and im.size_c==3
        assert im.output_axes=="YXS" and im.output_shape==(3,4,3)


@requires_schema
def test_label_background_metadata_is_caller_policy_not_a_tiff_tag(tmp_path):
    a=np.full((19,23),17,np.uint32); a[2:8,3:9]=4_000_000_000
    path=tmp_path/'label.ome.tif'
    with LabelImage.from_array(a,pixel_size=SIZE,background_label=17) as im:
        assert im.background_label==17
        OMETiffWriter(path,compression="Deflate",tile_size=16).write(im)
    with OMETiffLabelReader(path,background_label=17) as labels:
        np.testing.assert_array_equal(labels.asarray(),a)
        assert labels.background_label==17


@requires_schema
def test_nondeterministic_provider_fails_verification_not_silently_published(tmp_path):
    class DriftingSource(CoordinateSource):
        def _read_region(self,*args,**kwargs):
            data=super()._read_region(*args,**kwargs)
            return data+np.uint32(len(self.calls))
    path=tmp_path/'old.ome.tif'; path.write_bytes(b'preserved')
    with MultichannelImage(DriftingSource()) as im:
        with pytest.raises(ValueError,match="verification failed"):
            OMETiffWriter(path,compression="Deflate",pyramid_levels=0,tile_size=16).write(im)
        assert im.is_open
    assert path.read_bytes()==b'preserved'
    assert not list(tmp_path.glob('.omeify-*'))


@requires_schema
def test_series_descriptor_does_not_resurrect_closed_source(tmp_path):
    im=MultichannelImage.from_array(np.zeros((2,17,19),np.uint16),pixel_size=SIZE)
    series=OMEImageSeries.from_image("Borrowed",im)
    im.close(); im.open()
    with pytest.raises(RuntimeError,match="earlier"):
        OMEMultiSeriesWriter(tmp_path/'bad.ome.tif',compression="Deflate").write((series,))
    im.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("factory", [ArraySource, MultichannelImage.from_array])
def test_masked_arrays_require_explicit_upstream_policy(factory):
    data = np.ma.array(np.ones((3, 4), np.float32), mask=False)
    data.mask[1, 1] = True
    with pytest.raises(TypeError, match="unmasked"):
        factory(data, axes="YX")


def test_masked_provider_output_is_not_silently_unmasked():
    class MaskedSource(CoordinateSource):
        def _read_region(self, *args, **kwargs):
            data = super()._read_region(*args, **kwargs)
            return np.ma.array(data, mask=False)

    with MultichannelImage(MaskedSource()) as image:
        with pytest.raises(TypeError, match="unmasked"):
            image.read_region(0, 3, 0, 4)


@pytest.mark.parametrize("series", [True, np.bool_(False), 0.5, "0", -1])
def test_tiff_backend_rejects_noninteger_or_negative_series(series):
    with pytest.raises((TypeError, ValueError)):
        OMETiffSource("not-opened.ome.tif", series=series)


@pytest.mark.parametrize("background", [True, np.bool_(False), 0.5, "0"])
def test_tiff_backend_rejects_noninteger_background(background):
    with pytest.raises(TypeError, match="integer"):
        OMETiffSource("not-opened.ome.tif", image_type="label", background_label=background)


def test_generic_multichannel_facade_retains_rgb_sample_identity():
    with MultichannelImage(CoordinateSource("rgb")) as image:
        assert image.is_rgb
        assert image.sample_names == ("Red", "Green", "Blue")
        assert image.logical_channel_count == 1
        assert image.get_by_name("RGB").read_region(0, 3, 0, 4).shape == (3, 4, 3)
