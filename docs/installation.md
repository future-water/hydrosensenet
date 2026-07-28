# Installation

hydrosensenet requires Python >= 3.9.

```bash
pip install hydrosensenet
```

The core install covers network design, evaluation, and I/O.

## Optional extras

```bash
pip install "hydrosensenet[viz]"   # map plotting (matplotlib, cartopy)
pip install "hydrosensenet[nwm]"   # NWM download stack (fsspec, dask, pynhd, s3fs, zarr)
```

Features that need extras tell you which one to install when you call
them — for example `NetworkDesignResult.plot()` raises an error
suggesting `hydrosensenet[viz]` if matplotlib/cartopy are missing.

## Development install

```bash
git clone https://github.com/future-water/hydrosensenet.git
cd hydrosensenet
pip install -e ".[dev]"   # editable install with tests, linting, build tooling
pytest                    # run the test suite
```

To build this documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```
