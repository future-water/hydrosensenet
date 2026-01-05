"""Performance metrics for sensor network evaluation."""

import numpy as np
from typing import Tuple, Dict, Union


def calculate_performance_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate R-squared, NSE, and normalized NSE metrics."""
    # Calculate sum of squared residuals
    ss_res = np.sum((true_values - pred_values) ** 2, axis=0)

    # Calculate total sum of squares
    ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2, axis=0)

    # Calculate metrics with proper handling of division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        r_squared = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nse = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nnse = 1 / (2 - nse)

    return r_squared, nse, nnse


def reconstruction_error(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    metric: str = "rmse"
) -> Union[float, np.ndarray]:
    """Calculate reconstruction error."""
    if metric == "rmse":
        return np.sqrt(np.mean((true_values - pred_values) ** 2, axis=0))
    elif metric == "mae":
        return np.mean(np.abs(true_values - pred_values), axis=0)
    elif metric == "relative":
        return (
            np.linalg.norm(pred_values - true_values, 'fro') /
            np.linalg.norm(true_values, 'fro')
        )
    elif metric == "frobenius":
        return np.linalg.norm(pred_values - true_values, 'fro')
    else:
        raise ValueError(f"Unknown metric: {metric}")


def reconstruction_evaluation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    sensor_locations: np.ndarray,
    n_sensors: int,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Evaluate reconstruction performance for given sensor locations.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape (n_train_timesteps, n_locations).
    X_test : np.ndarray
        Testing data, shape (n_test_timesteps, n_locations).
    sensor_locations : np.ndarray
        Array of sensor location indices.
    n_sensors : int
        Number of sensors to use for evaluation.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'X_test_selected': Selected sensor data
        - 'X_test_reconstructed': Reconstructed full field
        - 'selected_sensors': Indices of selected sensors
        - 'non_selected_sensors': Indices of non-selected sensors
        - 'rmse': RMSE per location
        - 'relative_error': Overall relative error
    """
    # Convert to numpy arrays if pandas DataFrames
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    N_locations = X_test.shape[1]
    all_locations = np.arange(N_locations)

    # Select top n_sensors
    selected_sensors = sensor_locations[:n_sensors]
    non_selected_sensors = np.setdiff1d(all_locations, selected_sensors)

    if verbose:
        print(f"      - Extracting data for {n_sensors} selected sensors...")

    # Extract selected sensor data
    X_train_selected = X_train[:, selected_sensors]  # (n_train, n_sensors)
    X_test_selected = X_test[:, selected_sensors]    # (n_test, n_sensors)

    # Reconstruct full field using least squares
    # FAST METHOD: Avoid creating huge (n_train × n_test) intermediate matrix
    # Instead solve a small (n_sensors × n_sensors) system
    # This is 100-1000x faster for large datasets!

    if verbose:
        print(f"      - Computing Gram matrix ({n_sensors} × {n_sensors})...")

    # G = S^T S  (n_sensors × n_sensors)
    G = X_train_selected.T @ X_train_selected

    if verbose:
        print(f"      - Solving linear system for reconstruction...")

    # Solve G Y = T^T  => Y (n_sensors × n_test)
    # Use lstsq for robustness - handles singular/rank-deficient matrices
    Y, residuals, rank, s = np.linalg.lstsq(G, X_test_selected.T, rcond=None)

    if verbose and rank < n_sensors:
        print(f"      ⚠ Warning: Gram matrix is rank-deficient (rank={rank}, expected={n_sensors})")
        print(f"        This suggests some sensors provide redundant information.")

    if verbose:
        print(f"      - Reconstructing full field ({X_test.shape[0]:,} × {N_locations:,})...")

    # Precompute M = S^T X_train  (n_sensors × n_locations)
    M = X_train_selected.T @ X_train

    # X_recon = Y^T M  (n_test × n_locations)
    X_test_reconstructed = Y.T @ M

    # Ensure non-negative for physical variables
    X_test_reconstructed = np.maximum(X_test_reconstructed, 1e-10)

    if verbose:
        print(f"      - Calculating error metrics...")

    # Calculate squared differences ONCE (avoid redundant computation)
    squared_diff = (X_test - X_test_reconstructed) ** 2

    # Calculate RMSE
    rmse = np.sqrt(np.mean(squared_diff, axis=0))

    # Calculate relative error
    relative_error = (
        np.linalg.norm(X_test_reconstructed - X_test, 'fro') /
        np.linalg.norm(X_test, 'fro')
    )

    # Calculate NSE and NNSE per location (reuse squared_diff)
    ss_res = np.sum(squared_diff, axis=0)
    ss_tot = np.sum((X_test - np.mean(X_test, axis=0)) ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        nse = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nnse = np.where((2 - nse) != 0, 1 / (2 - nse), np.nan)

    return {
        'X_test_selected': X_test_selected,
        'X_test_reconstructed': X_test_reconstructed,
        'selected_sensors': selected_sensors,
        'non_selected_sensors': non_selected_sensors,
        'rmse': rmse,
        'relative_error': relative_error,
        'nse': nse,
        'nnse': nnse
    }
