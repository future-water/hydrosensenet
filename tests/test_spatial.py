"""Tests for spatial weight calculation."""
import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from hydrosensenet import calculate_spatial_weights


def _locations(points, gauge_ids=None):
    data = {"gauge_id": gauge_ids} if gauge_ids is not None else {}
    return gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")


def test_array_source_passthrough():
    locations = _locations([Point(0, 0), Point(1, 1)])
    weights = calculate_spatial_weights(locations, np.array([2.0, 5.0]))
    np.testing.assert_allclose(weights, [2.0, 5.0])


def test_array_length_mismatch_raises():
    locations = _locations([Point(0, 0), Point(1, 1)])
    with pytest.raises(ValueError, match="length"):
        calculate_spatial_weights(locations, np.array([1.0, 2.0, 3.0]))


def test_dict_source_with_fill_value():
    locations = _locations(
        [Point(0, 0), Point(1, 1), Point(2, 2)], gauge_ids=["g1", "g2", "g3"]
    )
    weights = calculate_spatial_weights(
        locations, {"g1": 2.0, "g3": 5.0}, fill_value=0.0
    )
    np.testing.assert_allclose(weights, [2.0, 0.0, 5.0])


def test_polygon_source_assigns_by_intersection():
    locations = _locations([Point(0.5, 0.5), Point(2.5, 2.5), Point(10, 10)])
    weight_gdf = gpd.GeoDataFrame(
        {"risk": [3.0, 7.0]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    weights = calculate_spatial_weights(
        locations, weight_gdf, weight_column="risk", fill_value=-1.0
    )
    np.testing.assert_allclose(weights, [3.0, 7.0, -1.0])


def test_polygon_source_requires_weight_column():
    locations = _locations([Point(0.5, 0.5)])
    weight_gdf = gpd.GeoDataFrame(
        {"risk": [3.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="weight_column"):
        calculate_spatial_weights(locations, weight_gdf)


def test_polygon_source_unknown_column_raises():
    locations = _locations([Point(0.5, 0.5)])
    weight_gdf = gpd.GeoDataFrame(
        {"risk": [3.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="not in weight data"):
        calculate_spatial_weights(locations, weight_gdf, weight_column="nope")


def test_unknown_aggregation_raises():
    locations = _locations([Point(0.5, 0.5)])
    weight_gdf = gpd.GeoDataFrame(
        {"risk": [3.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="aggregation"):
        calculate_spatial_weights(
            locations, weight_gdf, weight_column="risk", aggregation="median"
        )


def test_invalid_source_type_raises():
    locations = _locations([Point(0, 0)])
    with pytest.raises(TypeError, match="weight_source"):
        calculate_spatial_weights(locations, 42)


def test_align_to_reorders_weights():
    locations = _locations([Point(0, 0), Point(1, 1)], gauge_ids=["a", "b"])
    weights = calculate_spatial_weights(
        locations, np.array([1.0, 2.0]), align_to=["b", "a"]
    )
    np.testing.assert_allclose(weights, [2.0, 1.0])


def test_align_to_missing_id_uses_fill_value():
    locations = _locations([Point(0, 0), Point(1, 1)], gauge_ids=["a", "b"])
    weights = calculate_spatial_weights(
        locations, np.array([1.0, 2.0]), align_to=["b", "zzz"], fill_value=9.0
    )
    np.testing.assert_allclose(weights, [2.0, 9.0])


def test_normalize_min_max_scales_weights():
    locations = _locations([Point(0, 0), Point(1, 1), Point(2, 2)])
    weights = calculate_spatial_weights(
        locations, np.array([1.0, 3.0, 5.0]), normalize=True
    )
    np.testing.assert_allclose(weights, [0.0, 0.5, 1.0])


def test_polygon_source_duplicated_index_raises():
    locations = _locations([Point(0.5, 0.5), Point(2.5, 2.5)])
    locations.index = [0, 0]
    weight_gdf = gpd.GeoDataFrame(
        {"risk": [3.0, 7.0]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="reset_index"):
        calculate_spatial_weights(locations, weight_gdf, weight_column="risk")


def test_align_to_partial_mismatch_warns():
    locations = _locations([Point(0, 0), Point(1, 1)], gauge_ids=["a", "b"])
    with pytest.warns(UserWarning, match=r"1 of 2 align_to ids not found"):
        weights = calculate_spatial_weights(
            locations, np.array([1.0, 2.0]), align_to=["b", "zzz"], fill_value=9.0
        )
    np.testing.assert_allclose(weights, [2.0, 9.0])


def test_align_to_dtype_mismatch_warns_and_mentions_dtype():
    # Integer gauge ids vs string align_to ids: nothing matches, so all
    # entries silently became fill_value before the warning was added.
    locations = _locations([Point(0, 0), Point(1, 1)], gauge_ids=[1, 2])
    with pytest.warns(UserWarning, match=r"2 of 2 .*dtype mismatch"):
        weights = calculate_spatial_weights(
            locations, np.array([1.0, 2.0]), align_to=["1", "2"], fill_value=0.0
        )
    np.testing.assert_allclose(weights, [0.0, 0.0])


def test_align_to_full_match_does_not_warn():
    locations = _locations([Point(0, 0), Point(1, 1)], gauge_ids=["a", "b"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weights = calculate_spatial_weights(
            locations, np.array([1.0, 2.0]), align_to=["b", "a"]
        )
    np.testing.assert_allclose(weights, [2.0, 1.0])
