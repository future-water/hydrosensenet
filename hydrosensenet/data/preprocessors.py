"""Data preprocessing utilities for sensor network optimization."""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Tuple, Dict, Optional, List


def split_timeseries(
    data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
    train_frac: float = 0.7,
    filter_invalid: bool = True,
    return_mapping: bool = False
) -> Union[Tuple, Tuple[any, any, Dict]]:
    """Split time series data into training and testing sets."""
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if isinstance(data, np.ndarray):
        return _split_array(data, train_frac, filter_invalid, return_mapping)
    elif isinstance(data, pd.DataFrame):
        return _split_dataframe(data, train_frac, filter_invalid, return_mapping)
    elif isinstance(data, xr.Dataset):
        return _split_dataset(data, train_frac, filter_invalid, return_mapping)
    else:
        raise TypeError(
            f"data must be np.ndarray, pd.DataFrame, or xr.Dataset, "
            f"got {type(data)}"
        )


def _split_array(
    data: np.ndarray,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split numpy array."""
    n_train = int(train_frac * data.shape[0])
    if n_train == 0 or n_train == data.shape[0]:
        raise ValueError(
            f"train_frac={train_frac} with {data.shape[0]} timesteps yields "
            f"an empty train or test set"
        )
    train_data = data[:n_train, :]
    test_data = data[n_train:, :]

    if filter_invalid:
        # Filter columns with NaN/inf in training (no copy when all valid)
        finite_mask = np.isfinite(train_data).all(axis=0)
        if finite_mask.all():
            good_cols = np.arange(data.shape[1])
        else:
            good_cols = np.where(finite_mask)[0]
            train_data = train_data[:, good_cols]
            test_data = test_data[:, good_cols]

        if return_mapping:
            mapping = {
                "old_to_new": {old: new for new, old in enumerate(good_cols)},
                "good_cols": good_cols,
                "n_removed": data.shape[1] - len(good_cols)
            }
            return train_data, test_data, mapping

    if return_mapping:
        mapping = {
            "old_to_new": {i: i for i in range(data.shape[1])},
            "good_cols": np.arange(data.shape[1]),
            "n_removed": 0
        }
        return train_data, test_data, mapping

    return train_data, test_data


def _split_dataframe(
    data: pd.DataFrame,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split pandas DataFrame."""
    n_train = int(train_frac * len(data))
    if n_train == 0 or n_train == len(data):
        raise ValueError(
            f"train_frac={train_frac} with {len(data)} timesteps yields "
            f"an empty train or test set"
        )
    train_data = data.iloc[:n_train, :]
    test_data = data.iloc[n_train:, :]

    if filter_invalid:
        # Filter columns with NaN/inf in training
        finite_mask = np.isfinite(train_data.values).all(axis=0)
        good_cols = data.columns[finite_mask]

        train_data = train_data[good_cols]
        test_data = test_data[good_cols]

        if return_mapping:
            mapping = {
                "good_cols": good_cols,
                "removed_cols": data.columns[~finite_mask],
                "n_removed": (~finite_mask).sum()
            }
            return train_data, test_data, mapping

    if return_mapping:
        mapping = {"good_cols": data.columns, "removed_cols": [], "n_removed": 0}
        return train_data, test_data, mapping

    return train_data, test_data


def _split_dataset(
    data: xr.Dataset,
    train_frac: float,
    filter_invalid: bool,
    return_mapping: bool
):
    """Split xarray Dataset."""
    time_dim = "time"  # Assume time dimension is named "time"
    n_train = int(train_frac * len(data[time_dim]))

    train_data = data.isel({time_dim: slice(0, n_train)})
    test_data = data.isel({time_dim: slice(n_train, None)})

    # Filtering for xarray is more complex, skipping for now
    if filter_invalid:
        warnings.warn(
            "filter_invalid is not supported for xarray Datasets; "
            "returning unfiltered splits",
            UserWarning,
            stacklevel=3,
        )
    if return_mapping:
        mapping = {"n_removed": 0}
        return train_data, test_data, mapping

    return train_data, test_data


def prepare_matrix(
    data: Union[pd.DataFrame, xr.Dataset],
    variable: Optional[str] = None,
    drop_na: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert DataFrame or xarray Dataset to 2D matrix for optimization.

    Parameters
    ----------
    data : DataFrame or Dataset
        Input data.
        - DataFrame: columns are locations, rows are time
        - Dataset: will stack spatial dimensions
    variable : str, optional
        Variable name to extract from Dataset (required for Dataset input).
    drop_na : bool, default=True
        Drop locations with any NaN values.

    Returns
    -------
    matrix : np.ndarray
        2D array of shape (n_timesteps, n_locations).
    location_labels : list of str
        Labels for each location (column).

    Examples
    --------
    >>> # DataFrame input
    >>> df = pd.DataFrame(np.random.randn(365, 100))
    >>> matrix, labels = prepare_matrix(df)

    >>> # xarray Dataset input
    >>> ds = xr.Dataset(...)
    >>> matrix, labels = prepare_matrix(ds, variable="discharge")
    """
    if isinstance(data, pd.DataFrame):
        if drop_na:
            data = data.dropna(axis=1)
        matrix = data.values
        location_labels = list(data.columns)

    elif isinstance(data, xr.Dataset):
        if variable is None:
            # Try to find the first data variable
            variable = list(data.data_vars)[0]

        data_array = data[variable]

        # Stack spatial dimensions
        spatial_dims = [d for d in data_array.dims if d != 'time']

        if len(spatial_dims) == 0:
            raise ValueError("No spatial dimensions found")

        stacked = data_array.stack(location=spatial_dims)

        if drop_na:
            stacked = stacked.dropna(dim='location', how='any')

        matrix = stacked.values
        location_labels = [str(loc) for loc in stacked.location.values]

    else:
        raise TypeError(f"data must be DataFrame or Dataset, got {type(data)}")

    return matrix, location_labels


def filter_valid_data(
    data: np.ndarray,
    min_valid_fraction: float = 0.8,
    remove_zeros: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter locations based on data quality.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_timesteps, n_locations).
    min_valid_fraction : float, default=0.8
        Minimum fraction of valid (non-NaN) values required.
    remove_zeros : bool, default=False
        Remove locations with all zeros (dry streams).

    Returns
    -------
    filtered_data : np.ndarray
        Filtered data matrix.
    valid_indices : np.ndarray
        Indices of kept locations.
    """
    n_timesteps = data.shape[0]

    # Check valid fraction
    valid_fraction = np.isfinite(data).sum(axis=0) / n_timesteps
    valid_mask = valid_fraction >= min_valid_fraction

    # Optionally remove all-zero locations
    if remove_zeros:
        nonzero_mask = (data != 0).any(axis=0)
        valid_mask = valid_mask & nonzero_mask

    valid_indices = np.where(valid_mask)[0]
    filtered_data = data[:, valid_indices]

    print(
        f"Filtered data: kept {len(valid_indices)} of {data.shape[1]} locations "
        f"({100*len(valid_indices)/data.shape[1]:.1f}%)"
    )

    return filtered_data, valid_indices
