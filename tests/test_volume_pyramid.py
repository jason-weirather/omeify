"""Independent voxel-bin expectations, not writer-to-reader agreement alone."""
import itertools

import numpy as np
import pytest

from omeify.io._writer.pyramid import mean_downsample_2x
from omeify.io.volume_pyramid import plan_volume_pyramid, reduce_zyx


def expected_means(a, factors):
    shape = tuple((n+f-1)//f for n,f in zip(a.shape[:3],factors))
    out = np.empty((*shape,*a.shape[3:]), dtype=a.dtype)
    for z,y,x in itertools.product(*(range(n) for n in shape)):
        block = a[z*factors[0]:(z+1)*factors[0], y*factors[1]:(y+1)*factors[1],
                  x*factors[2]:(x+1)*factors[2]]
        mean = block.astype(np.float64).mean(axis=(0,1,2))
        out[z,y,x] = np.rint(mean) if a.dtype.kind in 'iu' else mean
    return out


@pytest.mark.parametrize('dtype',['uint8','uint16','uint32','int8','int16','int32','float32','float64'])
@pytest.mark.parametrize('factors',[(1,2,2),(2,2,2),(2,1,1),(2,1,2)])
def test_means_odd_bins(dtype,factors):
    a=np.arange(5*7*9).reshape(5,7,9)%97
    if dtype[0] in 'if':a=a-43
    a=a.astype(dtype)
    np.testing.assert_allclose(reduce_zyx(a,factors),expected_means(a,factors),rtol=1e-6)


def test_xyz_rounds_once_not_plane_by_plane():
    a=np.array([[[0,0],[0,2]],[[1,1],[1,3]]],np.uint16)
    assert reduce_zyx(a,(2,2,2)).item()==1
    # First rounding the two 2D means gives 0 and 2, but that must not be the algorithm.
    a=np.array([[[0,0],[0,1]],[[0,1],[1,2]]],np.uint16)
    assert reduce_zyx(a,(2,2,2)).item()==1


def test_nearest_labels_and_rgb_samples():
    a=np.arange(5*7*9*3,dtype=np.uint16).reshape(5,7,9,3)
    np.testing.assert_array_equal(reduce_zyx(a,(2,2,2),method='nearest'),a[::2,::2,::2])
    np.testing.assert_array_equal(reduce_zyx(a,(2,2,2)),expected_means(a,(2,2,2)))


def test_xy_identical_to_existing_arithmetic():
    a=np.random.default_rng(14).normal(size=(4,7,9)).astype('float32')
    got=reduce_zyx(a,(1,2,2))
    want=np.stack([mean_downsample_2x(p,(4,5)) for p in a])
    assert got.tobytes()==want.tobytes()


def test_float_overflow_and_nonfinite_policy():
    maximum=np.finfo('float64').max
    a=np.full((2,2,2),maximum)
    assert reduce_zyx(a,(2,2,2)).item()==maximum
    a[0,0,0]=np.inf
    assert np.isinf(reduce_zyx(a,(2,2,2))).all()
    a[0,0,1]=-np.inf
    assert np.isnan(reduce_zyx(a,(2,2,2))).all()


def test_plan_xy_xyz_and_anisotropic():
    p=plan_volume_pyramid((17,1025,1023),tile_size=128,axes='xy')
    assert all(s[0]==17 for s in p.shapes)
    assert p.shapes[-1]==(17,65,64)
    p=plan_volume_pyramid((17,65,63),tile_size=16,axes='xyz',levels=2)
    assert p.shapes==((17,65,63),(9,33,32),(5,17,16))
    p=plan_volume_pyramid((17,65,63),tile_size=16,axes='auto',levels=4,spacing=(4.,.5,.5))
    assert p.steps[:3]==((1,2,2),)*3
    assert p.steps[3]==(2,2,2)
    with pytest.raises(ValueError,match='calibrated'):
        plan_volume_pyramid((9,32,32),axes='auto')
    with pytest.raises(ValueError,match='cannot reduce Z'):
        plan_volume_pyramid((9,32,32),axes='xy',factors=[(2,2,2)])


def test_anisotropy_planning_ignores_already_singleton_axes():
    plan=plan_volume_pyramid((1,128,128),axes='auto',spacing=(.01,1,1),tile_size=16)
    assert plan.shapes[-1]==(1,16,16)
    assert plan.steps==((1,2,2),)*3
