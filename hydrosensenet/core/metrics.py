"""Performance metrics for sensor network evaluation."""

import numpy as np
from typing import Tuple, Dict, Union


def calculate_performance_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate per-location NSE and normalized NSE (NNSE) skill scores.

    Parameters
    ----------
    true_values, pred_values : np.ndarray
        Arrays of shape ``(n_timesteps, n_locations)``.

    Returns
    -------
    r_squared : np.ndarray
        Coefficient of determination per location. Computed with the
        NSE formula (1 - SSE/SST against the observed mean), so it is
        numerically identical to ``nse``.
    nse : np.ndarray
        Nash-Sutcliffe efficiency per location, in ``(-inf, 1]``.
        ``NaN`` where the true series is constant (zero variance).
    nnse : np.ndarray
        Normalized NSE, ``1 / (2 - NSE)``, mapping ``(-inf, 1]`` to
        ``(0, 1]`` — convenient for mapping and aggregation.
    """
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

    # Reconstruct the full field with the least-squares operator learned on
    # training data: B = argmin ||S_train B - X_train||_F, applied as
    # X_recon = S_test @ B. The tall system is solved directly (SVD-based
    # lstsq): forming the Gram matrix S^T S instead would square the
    # condition number and lose half the significant digits for
    # ill-conditioned sensor subsets.
    if verbose:
        print(f"      - Solving least-squares reconstruction operator...")

    B, _, rank, _ = np.linalg.lstsq(X_train_selected, X_train, rcond=None)

    if verbose and rank < n_sensors:
        print(f"      ⚠ Warning: sensor matrix is rank-deficient (rank={rank}, expected={n_sensors})")
        print(f"        This suggests some sensors provide redundant information.")

    if verbose:
        print(f"      - Reconstructing full field ({X_test.shape[0]:,} × {N_locations:,})...")

    # X_recon = S_test @ B  (n_test × n_locations)
    X_test_reconstructed = X_test_selected @ B

    # Clamp to the physical range only when the data itself is non-negative
    # (e.g. streamflow); general anomaly fields keep negative values.
    if np.min(X_train) >= 0 and np.min(X_test) >= 0:
        X_test_reconstructed = np.maximum(X_test_reconstructed, 0.0)

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
