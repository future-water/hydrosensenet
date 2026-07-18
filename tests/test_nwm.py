"""Network-free tests for NWM data utilities (pure logic + monkeypatched deps)."""
import contextlib
import random
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrosensenet.data import nwm
from hydrosensenet.data.nwm import (
    NWMDataLoader,
    _area_sqkm,
    _chunked,
    _select_zarr_store,
    _warn_missing_comids,
)


# ---------------------------------------------------------------------------
# _chunked
# ---------------------------------------------------------------------------

def test_chunked_exact_multiple():
    chunks = list(_chunked(list(range(6)), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5]]


def test_chunked_with_remainder():
    chunks = list(_chunked(list(range(2500)), 1000))
    assert [len(c) for c in chunks] == [1000, 1000, 500]
    # Order and content preserved
    assert [x for c in chunks for x in c] == list(range(2500))


def test_chunked_size_larger_than_sequence():
    assert list(_chunked([1, 2], 10)) == [[1, 2]]


def test_chunked_empty_sequence():
    assert list(_chunked([], 5)) == []


def test_chunked_invalid_size_raises():
    with pytest.raises(ValueError, match="Chunk size"):
        list(_chunked([1, 2, 3], 0))


# ---------------------------------------------------------------------------
# _warn_missing_comids
# ---------------------------------------------------------------------------

def test_warn_missing_comids_all_present(recwarn):
    missing = _warn_missing_comids([1, 2, 3], [3, 2, 1])
    assert missing == set()
    assert len(recwarn) == 0


def test_warn_missing_comids_warns_with_count():
    with pytest.warns(UserWarning, match="2 of 4 requested COMIDs"):
        missing = _warn_missing_comids([1, 2, 3, 4], [2, 4])
    assert missing == {1, 3}


# ---------------------------------------------------------------------------
# _select_zarr_store
# ---------------------------------------------------------------------------

def test_select_zarr_store_by_name(recwarn):
    listing = [
        "bucket/CONUS/zarr/gwout.zarr",
        "bucket/CONUS/zarr/chrtout.zarr",
        "bucket/CONUS/zarr/ldasout.zarr",
    ]
    assert _select_zarr_store(listing) == "bucket/CONUS/zarr/chrtout.zarr"
    assert len(recwarn) == 0


def test_select_zarr_store_case_insensitive():
    listing = ["bucket/zarr/GWOUT.zarr", "bucket/zarr/CHRTOUT.zarr"]
    assert _select_zarr_store(listing) == "bucket/zarr/CHRTOUT.zarr"


def test_select_zarr_store_no_match_falls_back_with_warning():
    listing = ["bucket/zarr/a.zarr", "bucket/zarr/b.zarr", "bucket/zarr/c.zarr"]
    with pytest.warns(UserWarning, match="No entries matching"):
        selected = _select_zarr_store(listing)
    assert selected == "bucket/zarr/b.zarr"


def test_select_zarr_store_ambiguous_falls_back_with_warning():
    listing = ["bucket/zarr/chrtout_a.zarr", "bucket/zarr/chrtout_b.zarr"]
    with pytest.warns(UserWarning, match="2 entries matching"):
        selected = _select_zarr_store(listing)
    assert selected == "bucket/zarr/chrtout_b.zarr"


# ---------------------------------------------------------------------------
# _area_sqkm
# ---------------------------------------------------------------------------

def _one_degree_square_at_40n():
    from shapely.geometry import box

    # 1 deg x 1 deg square centered at 40N, 105W (within CONUS)
    return box(-105.5, 39.5, -104.5, 40.5)


def _expected_square_area_sqkm():
    # Spherical-earth area of the square: R^2 * dlon * (sin(lat2) - sin(lat1))
    r = 6371.0
    dlon = np.deg2rad(1.0)
    return r ** 2 * dlon * (np.sin(np.deg2rad(40.5)) - np.sin(np.deg2rad(39.5)))


def test_area_sqkm_matches_spherical_estimate():
    gpd = pytest.importorskip("geopandas")

    geoms = gpd.GeoSeries([_one_degree_square_at_40n()], crs="EPSG:4326")
    area = _area_sqkm(geoms).iloc[0]

    expected = _expected_square_area_sqkm()  # ~9575 sq km
    assert area == pytest.approx(expected, rel=0.02)
    # The naive degrees-based computation (deg^2 / 1e6 = 1e-6) is off by
    # roughly 10 orders of magnitude; make sure we are nowhere near it.
    assert area > 1000


def test_area_sqkm_accepts_plain_geometries_with_crs():
    pytest.importorskip("geopandas")

    area = _area_sqkm([_one_degree_square_at_40n()], crs="EPSG:4326").iloc[0]
    assert area == pytest.approx(_expected_square_area_sqkm(), rel=0.02)


def test_area_sqkm_without_crs_raises():
    pytest.importorskip("geopandas")

    with pytest.raises(ValueError, match="no CRS"):
        _area_sqkm([_one_degree_square_at_40n()])


# ---------------------------------------------------------------------------
# filter_by_huc validation
# ---------------------------------------------------------------------------

def test_filter_by_huc_rejects_codes_longer_than_8_digits():
    loader = NWMDataLoader(use_dask=False)
    with pytest.raises(ValueError, match="longer than 8 digits"):
        loader.filter_by_huc("1204010501")


def test_filter_by_huc_rejects_huc12():
    loader = NWMDataLoader(use_dask=False)
    with pytest.raises(ValueError, match="HUC12"):
        loader.filter_by_huc("120401050101")


# ---------------------------------------------------------------------------
# Lazy Dask cluster
# ---------------------------------------------------------------------------

def test_init_does_not_start_dask_cluster():
    loader = NWMDataLoader(use_dask=True)
    assert loader.client is None
    assert loader.cluster is None


def test_setup_dask_without_dask_installed_disables_use_dask(monkeypatch):
    # Simulate a missing dask.distributed regardless of what the test
    # environment has installed: the lazy startup must fall back
    # gracefully instead of raising.
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.distributed", None)
    loader = NWMDataLoader(use_dask=True)
    loader._setup_dask()
    assert loader.use_dask is False
    assert loader.client is None
    assert loader.cluster is None


def test_setup_dask_noop_when_disabled():
    loader = NWMDataLoader(use_dask=False)
    loader._setup_dask()
    assert loader.client is None
    assert loader.cluster is None


def test_close_is_idempotent_and_safe_when_never_started():
    loader = NWMDataLoader(use_dask=True)
    loader.close()
    loader.close()
    assert loader.client is None
    assert loader.cluster is None


def test_close_shuts_down_client_and_cluster():
    closed = []

    class _Fake:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    loader = NWMDataLoader(use_dask=True)
    loader.client = _Fake("client")
    loader.cluster = _Fake("cluster")
    loader.close()
    assert closed == ["client", "cluster"]
    assert loader.client is None
    assert loader.cluster is None


def test_context_manager_closes_on_exit():
    with NWMDataLoader(use_dask=True) as loader:
        assert isinstance(loader, NWMDataLoader)
        closed = []
        loader.client = types.SimpleNamespace(close=lambda: closed.append("client"))
        loader.cluster = types.SimpleNamespace(close=lambda: closed.append("cluster"))
    assert closed == ["client", "cluster"]
    assert loader.client is None
    assert loader.cluster is None


# ---------------------------------------------------------------------------
# export_locations (with a fake pynhd module)
# ---------------------------------------------------------------------------

def _install_fake_pynhd(monkeypatch, queried, drop_comids=()):
    """Install a fake ``pynhd`` module whose nhdplus_l48 records SQL queries."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    def fake_nhdplus_l48(layer=None, sql=None):
        assert layer == "NHDFlowline_Network"
        assert sql is not None and sql.startswith("COMID IN (") and sql.endswith(")")
        ids = [int(x) for x in sql[len("COMID IN ("):-1].split(",")]
        queried.append(ids)
        kept = [i for i in ids if i not in set(drop_comids)]
        return gpd.GeoDataFrame(
            {"COMID": kept},
            geometry=[LineString([(c, 0), (c + 1, 1)]) for c in kept],
            crs="EPSG:4326",
        )

    fake_pynhd = types.ModuleType("pynhd")
    fake_pynhd.nhdplus_l48 = fake_nhdplus_l48
    monkeypatch.setitem(sys.modules, "pynhd", fake_pynhd)


def test_export_locations_queries_all_comids_in_chunks(tmp_path, monkeypatch):
    queried = []
    _install_fake_pynhd(monkeypatch, queried)

    comids = list(range(1, 2501))  # more than the old 1000-COMID truncation
    loader = NWMDataLoader(use_dask=False).filter_by_comids(comids)
    out = loader.export_locations(tmp_path / "locations.csv", format="csv")

    assert len(queried) == 3
    assert [len(q) for q in queried] == [1000, 1000, 500]
    assert sorted(c for q in queried for c in q) == comids

    df = pd.read_csv(out)
    assert sorted(df["comid"]) == comids


def test_export_locations_warns_on_missing_comids(tmp_path, monkeypatch):
    queried = []
    _install_fake_pynhd(monkeypatch, queried, drop_comids=(7, 8))

    loader = NWMDataLoader(use_dask=False).filter_by_comids(list(range(1, 11)))
    with pytest.warns(UserWarning, match="2 of 10 requested COMIDs"):
        out = loader.export_locations(tmp_path / "locations.csv", format="csv")

    df = pd.read_csv(out)
    assert sorted(df["comid"]) == [1, 2, 3, 4, 5, 6, 9, 10]


def test_export_locations_random_state_is_reproducible(tmp_path, monkeypatch):
    queried = []
    _install_fake_pynhd(monkeypatch, queried)

    comids = list(range(1, 101))
    expected = sorted(random.Random(42).sample(comids, 10))

    loader = NWMDataLoader(use_dask=False).filter_by_comids(comids)
    out = loader.export_locations(
        tmp_path / "sampled.csv", format="csv", sample_size=10, random_state=42
    )

    df = pd.read_csv(out)
    assert sorted(df["comid"]) == expected


def test_export_locations_without_comids_raises(tmp_path):
    loader = NWMDataLoader(use_dask=False)
    with pytest.raises(ValueError, match="No COMIDs"):
        loader.export_locations(tmp_path / "locations.csv")


# ---------------------------------------------------------------------------
# download_streamflow empty-result guard (with fake dask + in-memory dataset)
# ---------------------------------------------------------------------------

def _fake_dask_module():
    class _Config:
        @staticmethod
        def set(**kwargs):
            return contextlib.nullcontext()

    return types.SimpleNamespace(config=_Config)


def _make_loader_with_dataset(monkeypatch):
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    ds = xr.Dataset(
        {"streamflow": (("time", "feature_id"), np.ones((4, 3)))},
        coords={"time": times, "feature_id": [10, 20, 30]},
    )
    monkeypatch.setattr(nwm, "_import_nwm_deps", lambda: (None, _fake_dask_module()))
    loader = NWMDataLoader(use_dask=False)
    loader.ds = ds  # pre-populated so _open_dataset never touches the network
    return loader


def test_download_streamflow_no_matching_comids_raises(monkeypatch):
    loader = _make_loader_with_dataset(monkeypatch)
    loader.filter_by_comids([999])  # not in the dataset

    with pytest.raises(ValueError, match="No NWM data returned"):
        loader.download_streamflow("2020-01-01", "2020-01-02", resample=None)


def test_download_streamflow_returns_data_for_matching_comids(monkeypatch):
    loader = _make_loader_with_dataset(monkeypatch)
    loader.filter_by_comids([10, 20])

    df = loader.download_streamflow("2020-01-01", "2020-01-02", resample=None)
    assert sorted(df.columns) == [10, 20]
    assert len(df) == 4
