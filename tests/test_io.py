"""Round-trip tests for I/O utilities."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from hydrosensenet import (
    load_locations,
    load_streamflow,
    migrate_csv_to_parquet,
    save_locations,
    save_streamflow,
)


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
def locations_gdf():
    return gpd.GeoDataFrame(
        {"gauge_id": ["g1", "g2"]},
        geometry=[Point(-97.5, 30.2), Point(-96.8, 29.9)],
        crs="EPSG:4326",
    )


def test_streamflow_parquet_roundtrip(tmp_path, streamflow_df):
    path = save_streamflow(streamflow_df, tmp_path / "flow.parquet")
    loaded = load_streamflow(path)
    # Parquet does not preserve the DatetimeIndex freq attribute
    pd.testing.assert_frame_equal(loaded, streamflow_df, check_freq=False)


def test_streamflow_csv_roundtrip(tmp_path, streamflow_df):
    path = save_streamflow(streamflow_df, tmp_path / "flow.csv")
    loaded = load_streamflow(path)
    np.testing.assert_allclose(loaded.values, streamflow_df.values)
    assert list(loaded.columns) == list(streamflow_df.columns)


def test_save_streamflow_defaults_to_parquet(tmp_path, streamflow_df):
    path = save_streamflow(streamflow_df, tmp_path / "flow")
    assert path.suffix == ".parquet"
    assert path.exists()


def test_load_streamflow_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_streamflow(tmp_path / "missing.parquet")


def test_load_streamflow_column_selection(tmp_path, streamflow_df):
    path = save_streamflow(streamflow_df, tmp_path / "flow.parquet")
    loaded = load_streamflow(path, columns=["g1", "g3"])
    assert list(loaded.columns) == ["g1", "g3"]


def test_locations_parquet_roundtrip(tmp_path, locations_gdf):
    path = save_locations(locations_gdf, tmp_path / "gauges.parquet")
    loaded = load_locations(path)
    assert isinstance(loaded, gpd.GeoDataFrame)
    assert list(loaded["gauge_id"]) == ["g1", "g2"]
    np.testing.assert_allclose(loaded.geometry.x, locations_gdf.geometry.x)


def test_locations_geojson_roundtrip(tmp_path, locations_gdf):
    path = save_locations(locations_gdf, tmp_path / "gauges.geojson")
    loaded = load_locations(path)
    assert isinstance(loaded, gpd.GeoDataFrame)
    assert list(loaded["gauge_id"]) == ["g1", "g2"]


def test_locations_csv_extracts_coordinates(tmp_path, locations_gdf):
    path = save_locations(locations_gdf, tmp_path / "gauges.csv")
    loaded = load_locations(path)
    assert "longitude" in loaded.columns
    assert "latitude" in loaded.columns
    np.testing.assert_allclose(loaded["longitude"], locations_gdf.geometry.x)


def test_migrate_csv_to_parquet(tmp_path, streamflow_df):
    csv_path = tmp_path / "flow.csv"
    streamflow_df.to_csv(csv_path)

    parquet_path = migrate_csv_to_parquet(csv_path)

    assert parquet_path == tmp_path / "flow.parquet"
    loaded = pd.read_parquet(parquet_path)
    np.testing.assert_allclose(loaded.values, streamflow_df.values)
    assert csv_path.exists()  # not removed by default


def test_load_streamflow_csv_column_selection(tmp_path, streamflow_df):
    path = save_streamflow(streamflow_df, tmp_path / "flow.csv")
    loaded = load_streamflow(path, columns=["g3", "g1"])
    assert list(loaded.columns) == ["g3", "g1"]
    assert isinstance(loaded.index, pd.DatetimeIndex)
    np.testing.assert_allclose(loaded["g1"].values, streamflow_df["g1"].values)
    np.testing.assert_allclose(loaded["g3"].values, streamflow_df["g3"].values)


def test_load_locations_plain_parquet_fallback(tmp_path):
    df = pd.DataFrame({"gauge_id": ["g1", "g2"], "longitude": [-97.5, -96.8]})
    path = tmp_path / "plain.parquet"
    df.to_parquet(path)

    loaded = load_locations(path)
    assert not isinstance(loaded, gpd.GeoDataFrame)
    assert list(loaded["gauge_id"]) == ["g1", "g2"]


def test_load_locations_corrupted_parquet_raises_original_error(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"not a parquet file")

    with pytest.raises(pyarrow.lib.ArrowInvalid):
        load_locations(path)


def test_load_locations_non_valueerror_propagates(tmp_path, monkeypatch):
    # A plain parquet file so the old bare-except fallback would have
    # silently succeeded and masked the real error.
    df = pd.DataFrame({"gauge_id": ["g1"], "longitude": [-97.5]})
    path = tmp_path / "plain.parquet"
    df.to_parquet(path)

    def boom(*args, **kwargs):
        raise ImportError("pyarrow is broken")

    monkeypatch.setattr(gpd, "read_parquet", boom)
    with pytest.raises(ImportError, match="pyarrow is broken"):
        load_locations(path)


def test_unknown_explicit_format_raises(tmp_path, streamflow_df, locations_gdf):
    flow_path = save_streamflow(streamflow_df, tmp_path / "flow.parquet")
    loc_path = save_locations(locations_gdf, tmp_path / "gauges.parquet")

    with pytest.raises(ValueError, match="Unknown format"):
        save_streamflow(streamflow_df, tmp_path / "out.parquet", format="hdf5")
    with pytest.raises(ValueError, match="Unknown format"):
        load_streamflow(flow_path, format="hdf5")
    with pytest.raises(ValueError, match="Unknown format"):
        save_locations(locations_gdf, tmp_path / "out.parquet", format="hdf5")
    with pytest.raises(ValueError, match="Unknown format"):
        load_locations(loc_path, format="hdf5")
