# Contributing to hydrosensenet

Thank you for your interest in improving hydrosensenet. Contributions of all
kinds are welcome: bug reports, documentation fixes, new features, and
examples. By contributing, you agree that your work is licensed under the
project's BSD-3-Clause license.

## Development setup

```bash
git clone https://github.com/future-water/hydrosensenet.git
cd hydrosensenet
pip install -e ".[dev,docs]"
```

Run the test suite:

```bash
pytest
```

Build the documentation exactly as CI does (warnings are errors):

```bash
sphinx-build -W -b html docs docs/_build/html
```

### Optional extras

The core install stays lightweight. Heavier stacks are opt-in extras:

- `viz` — plotting support (matplotlib, cartopy)
- `nwm` — National Water Model ingestion (dask, zarr, s3fs, fsspec, pynhd,
  pygeohydro)

Install them only if your change touches those areas, for example
`pip install -e ".[dev,nwm]"`. Code in the core package must not import from
an extra at module level; keep such imports inside the functions that need
them so a plain install keeps working.

## Code style

- CI lints with flake8 restricted to error-class rules
  (`--select=E9,F63,F7,F82`): syntax errors, undefined names, and similar
  hard failures. Your change must pass this gate.
- Format code with `black` (configured in `pyproject.toml`, line length 100).
- Write docstrings in the numpydoc style (`Parameters`, `Returns`,
  `Examples` sections), matching the existing modules.

## Reporting bugs

Open an issue at
<https://github.com/future-water/hydrosensenet/issues> and include:

1. A minimal reproducible example — the smallest script or snippet that
   triggers the problem, using synthetic data or the bundled example basin
   where possible.
2. The full traceback or the incorrect output you observed.
3. Your environment: OS, Python version, and `hydrosensenet.__version__`.

Issues without a reproducible example are hard to act on and may be closed
until one is provided.

## Proposing changes

1. Fork the repository and create a topic branch from `main`.
2. Make your changes, keeping commits focused.
3. Add or update tests for any change in behavior. New features need tests
   that exercise them; bug fixes need a test that fails without the fix.
4. Keep documentation examples runnable. The docs build with `sphinx-build
   -W`, and code shown in the docs should execute against the bundled
   example dataset without extra downloads.
5. Run `pytest`, the flake8 gate, and the docs build locally.
6. Open a pull request against `main` describing what the change does and
   why. CI runs the lint gate, the test suite on Python 3.9 through 3.13,
   the docs build, and a package build; all must pass before review.

Questions about the design or scope of a change? Open an issue first to
discuss before writing code.
