"""Tests for the high-level SensorNetworkDesigner API."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from hydrosensenet import SensorNetworkDesigner


def _make_locations(n):
    return gpd.GeoDataFrame(
        {"id": range(n)},
        geometry=[Point(i, i) for i in range(n)],
        crs="EPSG:4326",
    )


def _rank3_positive_data(n_timesteps=200, n_locations=20, seed=1):
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.5, 1.5, size=(n_timesteps, 3))
    V = rng.uniform(0.5, 2.0, size=(3, n_locations))
    return U @ V


def test_designer_initialization():
    n_locations, n_timesteps = 10, 100
    rng = np.random.default_rng(0)
    streamflow_data = rng.random((n_timesteps, n_locations))
    locations_gdf = _make_locations(n_locations)
    labels = [f"gauge_{i}" for i in range(n_locations)]

    designer = SensorNetworkDesigner(
        streamflow_data=streamflow_data,
        locations=locations_gdf,
        location_labels=labels,
    )

    assert designer.streamflow_data.shape == (n_timesteps, n_locations)
    assert designer.location_labels == labels


def test_dimension_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="locations"):
        SensorNetworkDesigner(
            streamflow_data=rng.random((100, 10)),
            locations=_make_locations(5),
        )


def test_default_labels_are_generated():
    rng = np.random.default_rng(0)
    designer = SensorNetworkDesigner(
        streamflow_data=rng.random((50, 4)),
        locations=_make_locations(4),
    )
    assert designer.location_labels == ["0", "1", "2", "3"]


def test_design_network_without_evaluation():
    rng = np.random.default_rng(0)
    designer = SensorNetworkDesigner(
        streamflow_data=rng.random((100, 10)),
        locations=_make_locations(10),
    )

    result = designer.design_network(n_sensors=5, evaluate=False, verbose=False)

    assert len(result.selected_indices) == 5
    assert result.performance_metrics is None
    assert len(result.locations) == 5


def test_design_network_recovers_low_rank_field():
    """End-to-end: 3 sensors must near-exactly reconstruct a rank-3 field."""
    X = _rank3_positive_data()
    designer = SensorNetworkDesigner(
        streamflow_data=X,
        locations=_make_locations(X.shape[1]),
        location_labels=[f"gauge_{i}" for i in range(X.shape[1])],
    )

    result = designer.design_network(n_sensors=3, evaluate=True, verbose=False)

    nse = result.performance_metrics["nse"]
    assert np.nanmedian(nse) > 0.99
    eval_results = result.performance_metrics["eval_results"]
    assert eval_results["relative_error"] < 1e-6
    # Selected labels follow selected indices
    expected_labels = [f"gauge_{i}" for i in result.selected_indices]
    assert result.location_labels == expected_labels


def test_existing_sensors_are_kept():
    X = _rank3_positive_data()
    designer = SensorNetworkDesigner(
        streamflow_data=X,
        locations=_make_locations(X.shape[1]),
    )

    result = designer.design_network(
        n_sensors=5, existing_sensors=[2, 7], evaluate=False, verbose=False
    )

    assert list(result.selected_indices[:2]) == [2, 7]
    assert len(result.selected_indices) == 5


def test_export_csv(tmp_path):
    X = _rank3_positive_data()
    designer = SensorNetworkDesigner(
        streamflow_data=X,
        locations=_make_locations(X.shape[1]),
    )
    result = designer.design_network(n_sensors=3, evaluate=False, verbose=False)

    out = tmp_path / "sensors.csv"
    result.export(out)

    df = pd.read_csv(out)
    assert len(df) == 3
    assert "sensor_rank" in df.columns
    assert "longitude" in df.columns
    assert "latitude" in df.columns


def test_get_dataframe():
    X = _rank3_positive_data()
    designer = SensorNetworkDesigner(
        streamflow_data=X,
        locations=_make_locations(X.shape[1]),
    )
    result = designer.design_network(n_sensors=3, evaluate=True, verbose=False)

    df = result.get_dataframe()

    assert len(df) == 3
    assert list(df["sensor_rank"]) == [1, 2, 3]
    assert "nse" in df.columns
    assert "longitude" in df.columns


def _nan_poisoned_data(n_timesteps=200, n_locations=8, dominant=5, nan_col=2, seed=3):
    """Positive data with a dominant signal column and one NaN-poisoned column."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(1, 2, size=(n_timesteps, n_locations))
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    X[:, dominant] = 100.0 * (2 + np.sin(t))
    X[3, nan_col] = np.nan
    return X


def test_nan_column_indices_map_back_to_original_space():
    """Regression: dropped NaN columns must not shift reported sensor identity."""
    X = _nan_poisoned_data()
    designer = SensorNetworkDesigner(
        X, _make_locations(8), [f"g{i}" for i in range(8)]
    )

    with pytest.warns(UserWarning, match="NaN"):
        result = designer.design_network(n_sensors=1, evaluate=True, verbose=False)

    # Column 5 dominates; with column 2 dropped, the buggy filtered-space
    # index would report g4. Correct behavior reports g5.
    assert result.selected_indices[0] == 5
    assert result.location_labels == ["g5"]
    assert result.locations.iloc[0]["id"] == 5

    # Per-location metrics keep original length, NaN at the dropped column
    nse = result.performance_metrics["nse"]
    assert len(nse) == 8
    assert np.isnan(nse[2])
    assert not np.isnan(nse[5])


def test_nan_column_accepts_original_length_weights():
    X = _nan_poisoned_data()
    designer = SensorNetworkDesigner(X, _make_locations(8))

    weights = np.ones(8)  # original length must be accepted
    with pytest.warns(UserWarning, match="NaN"):
        result = designer.design_network(
            n_sensors=2, weights=weights, evaluate=True, verbose=False
        )
    assert 2 not in result.selected_indices  # dropped column never selected


def test_existing_sensor_in_original_space_with_nan_column():
    X = _nan_poisoned_data()
    designer = SensorNetworkDesigner(X, _make_locations(8))

    with pytest.warns(UserWarning, match="NaN"):
        result = designer.design_network(
            n_sensors=2, existing_sensors=[6], evaluate=True, verbose=False
        )
    assert result.selected_indices[0] == 6


def test_existing_sensor_on_dropped_column_raises():
    X = _nan_poisoned_data(nan_col=1)
    designer = SensorNetworkDesigner(X, _make_locations(8))

    with pytest.warns(UserWarning, match="NaN"):
        with pytest.raises(ValueError, match="existing_sensors"):
            designer.design_network(
                n_sensors=2, existing_sensors=[1], evaluate=True, verbose=False
            )


def test_evaluate_false_filters_nan_consistently():
    """evaluate=False must not crash on NaN data and must use original indices."""
    X = _nan_poisoned_data()
    designer = SensorNetworkDesigner(X, _make_locations(8))

    with pytest.warns(UserWarning, match="NaN"):
        result = designer.design_network(n_sensors=1, evaluate=False, verbose=False)

    assert result.selected_indices[0] == 5
    assert result.performance_metrics is None


def test_too_many_sensors_for_valid_columns_raises():
    X = _nan_poisoned_data()
    designer = SensorNetworkDesigner(X, _make_locations(8))

    with pytest.warns(UserWarning, match="NaN"):
        with pytest.raises(ValueError, match="valid"):
            designer.design_network(n_sensors=8, evaluate=True, verbose=False)


def test_dataframe_input_infers_labels():
    X = _rank3_positive_data(n_locations=5)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    designer = SensorNetworkDesigner(df, _make_locations(5))
    assert designer.location_labels == ["a", "b", "c", "d", "e"]
    assert isinstance(designer.streamflow_data, np.ndarray)
