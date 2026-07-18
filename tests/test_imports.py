"""Import-surface tests: public API present, heavy optional deps stay lazy."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_HAS_VIZ = importlib.util.find_spec("cartopy") is not None
_HAS_NWM = importlib.util.find_spec("fsspec") is not None and (
    importlib.util.find_spec("dask") is not None
)


def test_version_is_exposed():
    import hydrosensenet
    assert isinstance(hydrosensenet.__version__, str)


def test_public_api_symbols_exist():
    import hydrosensenet
    for name in hydrosensenet.__all__:
        assert getattr(hydrosensenet, name) is not None


def test_core_imports():
    from hydrosensenet.core import algorithms, metrics
    assert algorithms is not None
    assert metrics is not None


def test_data_imports():
    from hydrosensenet.data import loaders, preprocessors
    assert loaders is not None
    assert preprocessors is not None


def test_spatial_imports():
    from hydrosensenet.spatial import weights
    assert weights is not None


def test_io_imports():
    from hydrosensenet import io
    assert io is not None


def test_import_does_not_pull_heavy_optional_deps():
    """`import hydrosensenet` must not import viz/nwm/legacy-only deps.

    Runs in a subprocess so imports from other tests can't pollute the check.
    """
    code = (
        "import sys\n"
        "import hydrosensenet\n"
        "heavy = [m for m in ('cartopy', 'matplotlib', 'fsspec', 'dask', 'sklearn')"
        " if m in sys.modules]\n"
        "assert not heavy, f'heavy modules imported eagerly: {heavy}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_unknown_attribute_raises():
    import hydrosensenet
    with pytest.raises(AttributeError):
        hydrosensenet.does_not_exist


def _reset_lazy_module(name):
    """Force the next attribute access to go through __getattr__ again."""
    import hydrosensenet
    hydrosensenet.__dict__.pop(name, None)
    sys.modules.pop(f"hydrosensenet.{name}", None)


@pytest.mark.skipif(_HAS_VIZ, reason="requires an environment without cartopy")
def test_legacy_attr_warns_and_hints_viz_extra():
    """Without viz deps, legacy access must warn and raise the install hint."""
    import hydrosensenet
    _reset_lazy_module("sensor_network_utils")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        with pytest.raises(ImportError, match=r"hydrosensenet\[viz\]"):
            hydrosensenet.load_data


def test_glofas_legacy_module_resolves_lazily():
    """glofas_processing_utils needs only core deps, so it must load anywhere."""
    import hydrosensenet
    _reset_lazy_module("glofas_processing_utils")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        module = hydrosensenet.glofas_processing_utils
    assert hasattr(module, "load_boundary_shapefile")


@pytest.mark.skipif(_HAS_VIZ, reason="requires an environment without cartopy")
def test_plot_raises_viz_install_hint():
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    from hydrosensenet import NetworkDesignResult

    result = NetworkDesignResult(
        selected_indices=np.array([0]),
        locations=gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326"),
        location_labels=["g0"],
        performance_metrics=None,
        n_sensors=1,
        weights=None,
    )
    with pytest.raises(ImportError, match=r"hydrosensenet\[viz\]"):
        result.plot()


@pytest.mark.skipif(_HAS_NWM, reason="requires an environment without fsspec/dask")
def test_nwm_deps_raise_install_hint():
    from hydrosensenet.data.nwm import _import_nwm_deps

    with pytest.raises(ImportError, match=r"hydrosensenet\[nwm\]"):
        _import_nwm_deps()
