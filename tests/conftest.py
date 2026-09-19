"""Small owned test fixtures using the current public image construction API."""
from contextlib import ExitStack

import pytest

from omeify import LabelImage, MultichannelImage, RGBImage


@pytest.fixture
def image_factory():
    """Keep tiny array-image fixtures open for the duration of each test."""
    with ExitStack() as stack:
        def make(array, *, kind="multichannel", axes=None, **metadata):
            cls = {"multichannel": MultichannelImage, "rgb": RGBImage, "label": LabelImage}[kind]
            if axes is None:
                axes = "YXS" if kind == "rgb" else "CYX" if array.ndim == 3 else "YX"
            return stack.enter_context(cls.from_array(array, axes=axes, **metadata))
        yield make
