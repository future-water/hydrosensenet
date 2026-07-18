"""Tests for data preprocessing utilities."""
import numpy as np
import pandas as pd
import pytest

from hydrosensenet import filter_valid_data, prepare_matrix, split_timeseries


def test_split_array_proportions():
    data = np.arange(40, dtype=float).reshape(10, 4)
    train, test = split_timeseries(data, train_frac=0.7, filter_invalid=False)
    assert train.shape == (7, 4)
    assert test.shape == (3, 4)
    np.testing.assert_array_equal(np.vstack([train, test]), data)


def test_split_array_filters_nan_train_columns():
    data = np.ones((10, 4))
    data[2, 1] = np.nan  # NaN in training portion -> column dropped
    train, test, mapping = split_timeseries(
        data, train_frac=0.5, filter_invalid=True, return_mapping=True
    )
    assert train.shape == (5, 3)
    assert test.shape == (5, 3)
    np.testing.assert_array_equal(mapping["good_cols"], [0, 2, 3])
    assert mapping["n_removed"] == 1


def test_split_array_keeps_columns_with_nan_only_in_test():
    data = np.ones((10, 3))
    data[8, 1] = np.nan  # NaN only in test portion -> column kept
    train, test = split_timeseries(data, train_frac=0.5, filter_invalid=True)
    assert train.shape == (5, 3)
    assert test.shape == (5, 3)


def test_split_dataframe_filters_by_column_name():
    df = pd.DataFrame(np.ones((10, 3)), columns=["a", "b", "c"])
    df.loc[1, "b"] = np.nan
    train, test, mapping = split_timeseries(
        df, train_frac=0.5, filter_invalid=True, return_mapping=True
    )
    assert list(train.columns) == ["a", "c"]
    assert list(test.columns) == ["a", "c"]
    assert mapping["n_removed"] == 1


def test_split_invalid_type_raises():
    with pytest.raises(TypeError):
        split_timeseries([1, 2, 3])


def test_prepare_matrix_drops_nan_columns():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, np.nan, 3.0],
        "c": [4.0, 5.0, 6.0],
    })
    matrix, labels = prepare_matrix(df)
    assert matrix.shape == (3, 2)
    assert labels == ["a", "c"]


def test_prepare_matrix_keeps_nan_when_disabled():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    matrix, labels = prepare_matrix(df, drop_na=False)
    assert matrix.shape == (2, 2)
    assert labels == ["a", "b"]


def test_prepare_matrix_invalid_type_raises():
    with pytest.raises(TypeError):
        prepare_matrix(np.ones((3, 3)))


def test_filter_valid_data_by_fraction():
    data = np.ones((10, 3))
    data[:3, 1] = np.nan  # 70% valid < 80% threshold -> dropped
    filtered, kept = filter_valid_data(data, min_valid_fraction=0.8)
    assert filtered.shape == (10, 2)
    np.testing.assert_array_equal(kept, [0, 2])


def test_filter_valid_data_removes_zero_columns():
    data = np.ones((10, 3))
    data[:, 2] = 0.0
    filtered, kept = filter_valid_data(data, remove_zeros=True)
    assert filtered.shape == (10, 2)
    np.testing.assert_array_equal(kept, [0, 1])


def test_invalid_train_frac_raises():
    data = np.ones((10, 3))
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="train_frac"):
            split_timeseries(data, train_frac=bad)


def test_degenerate_split_raises():
    data = np.ones((10, 3))
    with pytest.raises(ValueError, match="empty"):
        split_timeseries(data, train_frac=0.05)  # int(0.5) == 0 train rows


def test_xarray_filter_invalid_warns():
    import xarray as xr

    ds = xr.Dataset(
        {"flow": (("time", "location"), np.ones((10, 3)))},
        coords={"time": np.arange(10), "location": np.arange(3)},
    )
    with pytest.warns(UserWarning, match="xarray"):
        train, test = split_timeseries(ds, train_frac=0.5, filter_invalid=True)
    assert len(train["time"]) == 5
