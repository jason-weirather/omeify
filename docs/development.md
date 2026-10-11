# Development and release checks

Omeify 0.22 has one **Python 3.11+** image-I/O installation: TIFF and Zarr, including
HTTP(S) reads. A missing codec, Zarr dependency, or OME XSD is a broken development
installation, not a reason to skip its regression tests. These commands run
locally and do not depend on GitHub Actions or another CI service.

The published **0.21.0** release is the Python 3.10 compatibility checkpoint.
Keep that version and the application's existing numerical requirements in legacy
environments. This checkout no longer contains a Python 3.10 test environment,
TOML backport, or a TIFF-only dependency mode.

## Ordinary development

Create a dedicated environment rather than upgrading an active analysis or model
inference environment. From the repository root, using an installed Python 3.11:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 .
python -m build
```

Use the corresponding interpreter and activation command on Windows. `pip check`
checks dependency metadata; it does not test pixel fidelity or source geometry.
The full suite exercises those separately with small deterministic fixtures.
`ruff check .` runs the complete configured style check; the narrower command
above checks fatal problems without making unrelated formatting debt a release gate.

## Required test environments

Run both TIFF and Zarr tests in **Python 3.11**, not just a syntax parser configured
for 3.11. Run the suite again on a newer supported Python with independent NGFF
models installed. Record the actual dependency versions and skipped-test reasons.

| Environment | Installation from this checkout | Purpose |
|---|---|---|
| Python 3.11, selected dependency floors | `python -m pip install -c constraints/python311-min.txt -e '.[dev]'` | Real baseline run, including NumPy 1.26.4, Zarr 3.1.3 and both storage backends |
| Python 3.11, resolver-selected dependencies | `python -m pip install -e '.[dev]'` | Check the current compatible dependency combination |
| Newer Python, 3.12+ | `python -m pip install -e '.[dev,interop]'` | Full ordinary suite plus independent metadata-model readback |

Use separate environments for these installations. The constraints file pins
selected image-I/O floors, not every transitive dependency and not a complete
scientific-application lock. The 3.11-specific TIFF/codec upper bounds remain
because removing Python 3.10 support does not make newer-Python dependencies
installable on 3.11. Newer runtimes use their own declared dependency branches.

For each environment:

```bash
python -m pip check
python -m pytest -q
omeify version --json
```

The 3.11 grammar regression guards against accidentally introducing syntax that
requires 3.12+. It is **not** a substitute for executing on the minimum interpreter.
Native `typing.Self` describes factories/context managers that actually return the
concrete class; semantic views that return another wrapper are not annotated that way.

## Optional integrations

`interop` contains independent NGFF models and requires Python 3.12+. It remains
a test dependency, not a runtime image-model framework. Absence of the optional
model package is reported as a skip. A broken installation or missing versioned
model inside an installed package must fail rather than masquerade as absence.

The optional `intelligence` integration retains its separate Python 3.13+
requirements. Ordinary image tests neither require nor call an external model.
The Java Bio-Formats readback test remains opt-in through `OMEIFY_BIOFORMATS_JAR`.
A skipped optional integration does not establish compatibility with that consumer.
Required Zarr, HTTP, TIFF codec, and OME XSD checks are never optional integrations.

## Installed-wheel check

Build from this checkout, install the resulting wheel **without format extras**
into a disposable environment, then run the smoke test outside the checkout.
This catches missing package resources and dependencies that editable installs
can conceal. The script uses real TIFF and Zarr encoders/readers and the OME XSD.

POSIX shell example, after `python -m build`:

```bash
CHECKOUT="$(pwd)"
TEMP_ENV="$(mktemp -d)"
python -m venv "$TEMP_ENV/venv"
"$TEMP_ENV/venv/bin/python" -m pip install "$CHECKOUT/dist/omeify-0.22.0-py3-none-any.whl"
"$TEMP_ENV/venv/bin/python" -m pip check
(cd "$TEMP_ENV" && "$TEMP_ENV/venv/bin/python" -I "$CHECKOUT/tests/installed_smoke.py")
```

Remove that temporary environment after reviewing the results. The smoke test
checks packaged schemas, plain-install dependency metadata, LZW/JPEG/labels,
Zarr Zstd readback, and TIFF/Zarr conversions with known coordinates. It does not
replace the full regression suite, the loopback HTTP tests, or testing intended
viewer versions and platforms. Build artifacts and test environments do not need
to be committed to the repository.
