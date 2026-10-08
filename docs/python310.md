# Python 3.10 compatibility checkpoint

The 0.21 line retains **Python >=3.10** for ordinary image I/O, inspection,
conversion, mutation, and cropping. It receives the hardening/report cleanup before
a separate future change raises the Python minimum. The 3.10 dependency branches,
`tomli` fallback, Ruff `py310` target, Docker base, and example environment remain.
This checkpoint does not declare 1.0 stability or promise indefinite maintenance of
Python 3.10 and its upstream dependencies.

The optional intelligence extra still has its separate Python 3.13+ requirements.
It is not needed by image I/O. A new display/inference environment should not force
an existing scientific execution environment to change its numerical or model stack.

## Review and test in a separate environment

Do not upgrade a working scientific environment in place merely to test this
checkpoint. Clone it first, or create a dedicated image-I/O test environment:

```bash
mamba create -n omeify310_env -c conda-forge python=3.10 pip
mamba activate omeify310_env
python -m pip install -c constraints/python310-numpy1.txt '.[dev]'
python -m pip check
python -m pytest -q
omeify version --json
```

Run those commands at this checkout. The constraints select a concrete **NumPy 1.x
image-I/O test lane**, not a complete TensorFlow/CellGate lock:

| Package | Version selected by the constraint file |
|---|---|
| NumPy | 1.26.4 |
| tifffile | 2024.8.30 |
| imagecodecs | 2024.6.1 |

The imagecodecs 2024.6.1 package documents testing with Python 3.10.11 and NumPy
1.26.4, and supplies CPython 3.10 wheels for the target desktop/server platforms.
That upstream evidence motivates the lane; the project's CI still has to verify
Omeify against it. See [the upstream package record](https://pypi.org/project/imagecodecs/2024.6.1/).
Do not use the yanked imagecodecs 2024.1.1 as a new lock simply because it is the
oldest version allowed by the existing dependency range.

For an application that needs different NumPy or codec versions, preserve its
own requirements and test their intersection with Omeify. `pip check` detects
metadata-level conflicts, not numerical correctness or model reproducibility.
Run that application's regression cases, including image reads and writer calls.
Python 3.10 syntax compatibility alone does not establish compatibility with an
entire existing scientific environment.

## What the automated checks cover

The workflow has a normal Python 3.10-3.14 core matrix, an additional Python 3.10 /
NumPy 1.x lane on Linux, macOS, and Windows, an optional-intelligence integration
lane, and fresh wheel-install checks on Python 3.10 and 3.13. Core jobs require
actual codecs and the local OME XSD before running tests. They do not count a missing
core dependency as an integration pass. The optional Bio-Formats/JVM readback still
requires its supplied JAR; it is not certified by the ordinary Python matrix.

The wheel job builds the source distribution and then the wheel through Hatch,
installs outside the checkout, and runs `tests/installed_smoke.py` under Python's
isolated mode. It checks packaged schema availability and real scalar/RGB/label
readback. Passing this job is separate from passing tests against a source checkout.
CI gates fatal Ruff diagnostics without changing the repository's broader configured
style rules; run `ruff check .` separately for the full style review.

## Publication sequence

Review and test the checkpoint first, then create a **new** 0.21.0 tag and publish
that exact commit/artifact to the public repository and PyPI. Do not move or replace
the existing 0.20.0 tag. This documentation and the version change do not themselves
create a tag, upload a package, or establish that a package-index release is available.

After publication and downstream regression testing, legacy applications can pin
`omeify==0.21.0` or a subsequently reviewed 0.21.x maintenance version, alongside
their existing numerical constraints. Keep any future Python-minimum change in a
separate commit/version. A future 3.11+ line must declare its real `Requires-Python`;
it should not install incompletely or provide a misleading empty compatibility extra.

Any later 0.21.x maintenance should stay focused on justified correctness fixes.
This is a deliberately preserved checkpoint, not a second implementation of Omeify
or a commitment to backport every future feature.
