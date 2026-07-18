"""Tests for universal data loaders."""
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from hydrosensenet import load_streamflow_data, prepare_gauge_locations


@pytest.fixture
def streamflow_df():
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(1, 100, size=(10, 3)),
        index=index,
        columns=["g1", "g2", "g3"],
    )


@pytest.fixture
def gauge_df():
    return pd.DataFrame(
        {
            "gauge_id": ["A", "B", "C"],
            "latitude": [30.2, 29.9, 31.1],
            "longitude": [-97.5, -96.8, -98.2],
        }
    )


# ---------------------------------------------------------------------------
# load_streamflow_data / _load_csv
# ---------------------------------------------------------------------------

def test_load_csv_unnamed_index(tmp_path, streamflow_df):
    path = tmp_path / "flow.csv"
    streamflow_df.to_csv(path)  # index written without a header name

    loaded = load_streamflow_data(path)

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.name is None
    assert list(loaded.columns) == ["g1", "g2", "g3"]
    np.testing.assert_allclose(loaded.values, streamflow_df.values)
    pd.testing.assert_index_equal(
        loaded.index, streamflow_df.index, check_names=False, exact=False
    )


def test_load_csv_time_alias_autodetected(tmp_path, streamflow_df):
    df = streamflow_df.reset_index().rename(columns={"index": "date"})
    path = tmp_path / "flow.csv"
    df.to_csv(path, index=False)

    loaded = load_streamflow_data(path)

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.name == "date"
    np.testing.assert_allclose(loaded.values, streamflow_df.values)


def test_load_csv_explicit_time_col(tmp_path, streamflow_df):
    df = streamflow_df.reset_index().rename(columns={"index": "obs_time"})
    path = tmp_path / "flow.csv"
    df.to_csv(path, index=False)

    loaded = load_streamflow_data(path, time_col="obs_time")

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.name == "obs_time"
    np.testing.assert_allclose(loaded.values, streamflow_df.values)


def test_load_csv_multi_file_concat(tmp_path, streamflow_df):
    first, second = streamflow_df.iloc[:6], streamflow_df.iloc[6:]
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    first.to_csv(paths[0])
    second.to_csv(paths[1])

    loaded = load_streamflow_data(paths)

    assert len(loaded) == len(streamflow_df)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    np.testing.assert_allclose(loaded.values, streamflow_df.values)


def test_load_csv_location_cols_selection(tmp_path, streamflow_df):
    path = tmp_path / "flow.csv"
    streamflow_df.to_csv(path)

    loaded = load_streamflow_data(path, location_cols=["g1", "g3"])

    assert list(loaded.columns) == ["g1", "g3"]
    np.testing.assert_allclose(loaded.values, streamflow_df[["g1", "g3"]].values)


def test_load_csv_non_datetime_index_kept(tmp_path):
    df = pd.DataFrame(
        {"g1": [1.0, 2.0, 3.0]}, index=["site-a", "site-b", "site-c"]
    )
    path = tmp_path / "flow.csv"
    df.to_csv(path)  # unnamed index column with unparseable dates

    loaded = load_streamflow_data(path)

    # parse_dates-style behavior: unparseable index stays as-is
    assert not isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.index) == ["site-a", "site-b", "site-c"]


# ---------------------------------------------------------------------------
# prepare_gauge_locations
# ---------------------------------------------------------------------------

def test_prepare_gauge_preserves_existing_gauge_id(gauge_df):
    gdf = prepare_gauge_locations(gauge_df)
    assert list(gdf["gauge_id"]) == ["A", "B", "C"]


def test_prepare_gauge_creates_gauge_id_when_absent(gauge_df):
    gdf = prepare_gauge_locations(gauge_df.drop(columns=["gauge_id"]))
    assert list(gdf["gauge_id"]) == [0, 1, 2]


def test_prepare_gauge_dropna_warns(gauge_df):
    gauge_df.loc[1, "latitude"] = np.nan

    with pytest.warns(UserWarning, match=r"Dropped 1 row.*alignment"):
        gdf = prepare_gauge_locations(gauge_df)

    assert len(gdf) == 2
    assert list(gdf["gauge_id"]) == ["A", "C"]


def test_prepare_gauge_no_warning_without_missing(gauge_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_gauge_locations(gauge_df)
    assert not any("Dropped" in str(w.message) for w in caught)


def test_prepare_gauge_geoparquet(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"gauge_id": ["g1", "g2"]},
        geometry=[Point(-97.5, 30.2), Point(-96.8, 29.9)],
        crs="EPSG:4326",
    )
    path = tmp_path / "gauges.parquet"
    gdf.to_parquet(path)

    loaded = prepare_gauge_locations(path)

    assert isinstance(loaded, gpd.GeoDataFrame)
    assert loaded.crs == "EPSG:4326"
    assert list(loaded["gauge_id"]) == ["g1", "g2"]
    np.testing.assert_allclose(loaded.geometry.x, gdf.geometry.x)


def test_prepare_gauge_geoparquet_normalizes_crs(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"gauge_id": ["g1"]},
        geometry=[Point(-97.5, 30.2)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    path = tmp_path / "gauges.parquet"
    gdf.to_parquet(path)

    loaded = prepare_gauge_locations(path, crs="EPSG:4326")

    assert loaded.crs == "EPSG:4326"
    np.testing.assert_allclose(loaded.geometry.x, [-97.5])
    np.testing.assert_allclose(loaded.geometry.y, [30.2])


@pytest.mark.parametrize("suffix", [".parquet", ".pq"])
def test_prepare_gauge_plain_parquet(tmp_path, gauge_df, suffix):
    path = tmp_path / f"gauges{suffix}"
    gauge_df.to_parquet(path)

    loaded = prepare_gauge_locations(path)

    assert isinstance(loaded, gpd.GeoDataFrame)
    assert loaded.crs == "EPSG:4326"
    assert list(loaded["gauge_id"]) == ["A", "B", "C"]
    np.testing.assert_allclose(loaded.geometry.x, gauge_df["longitude"])
    np.testing.assert_allclose(loaded.geometry.y, gauge_df["latitude"])
