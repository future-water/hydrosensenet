"""Tests for USGS gauge to NWM COMID matching."""
import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from hydrosensenet.data.usgs import match_usgs_to_nwm


def _gauges(staids, station_col="STAID"):
    return gpd.GeoDataFrame(
        {station_col: staids},
        geometry=[Point(i, i) for i in range(len(staids))],
        crs="EPSG:4326",
    )


def test_match_comids_aligned_with_gdf_rows():
    # Linkage row order differs from gauge row order, and only a subset of
    # COMIDs is available: comids[i] must still describe matched.iloc[i].
    gdf = _gauges(["001", "002", "003"])
    linkage = pd.DataFrame(
        {"STAID": ["003", "001", "002"], "COMID": [30, 10, 20]}
    )
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[10, 30], verbose=False
    )
    assert isinstance(comids, list)
    assert isinstance(matched, gpd.GeoDataFrame)
    assert len(comids) == len(matched) == 2
    assert "comid" in matched.columns
    expected = {"001": 10, "003": 30}
    for i, comid in enumerate(comids):
        row = matched.iloc[i]
        assert comid == row["comid"] == expected[row["STAID"]]


def test_match_handles_station_id_dtype_mismatch():
    # Integer station ids in the GeoDataFrame vs strings in the linkage
    gdf = _gauges([1001, 1002])
    linkage = pd.DataFrame({"STAID": ["1001", "1002"], "COMID": [10, 20]})
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[10, 20], verbose=False
    )
    assert comids == [10, 20]
    assert list(matched["comid"]) == [10, 20]


def test_match_duplicate_station_dedup_warns():
    gdf = _gauges(["001", "002"])
    linkage = pd.DataFrame(
        {"STAID": ["001", "001", "002"], "COMID": [10, 11, 20]}
    )
    with pytest.warns(UserWarning, match=r"1 duplicate station"):
        comids, matched = match_usgs_to_nwm(
            gdf, linkage, available_comids=[10, 11, 20], verbose=False
        )
    assert comids == [10, 20]
    assert list(matched["STAID"]) == ["001", "002"]
    assert list(matched["comid"]) == comids


def test_match_no_duplicates_does_not_warn():
    gdf = _gauges(["001", "002"])
    linkage = pd.DataFrame({"STAID": ["001", "002"], "COMID": [10, 20]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comids, matched = match_usgs_to_nwm(
            gdf, linkage, available_comids=[10, 20], verbose=False
        )
    assert comids == [10, 20]


def test_match_empty_linkage_returns_empty():
    gdf = _gauges(["001"])
    linkage = pd.DataFrame(columns=["STAID", "COMID"])
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[10], verbose=False
    )
    assert comids == []
    assert len(matched) == 0


def test_match_no_available_comids_returns_empty():
    gdf = _gauges(["001"])
    linkage = pd.DataFrame({"STAID": ["001"], "COMID": [10]})
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[999], verbose=False
    )
    assert comids == []
    assert len(matched) == 0


def test_match_no_station_column_returns_empty():
    gdf = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    linkage = pd.DataFrame({"STAID": ["001"], "COMID": [10]})
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[10], verbose=False
    )
    assert comids == []
    assert len(matched) == 0


def test_match_no_gdf_overlap_returns_empty():
    # Linkage stations that do not exist in the GeoDataFrame
    gdf = _gauges(["999"])
    linkage = pd.DataFrame({"STAID": ["001"], "COMID": [10]})
    comids, matched = match_usgs_to_nwm(
        gdf, linkage, available_comids=[10], verbose=False
    )
    assert comids == []
    assert len(matched) == 0
