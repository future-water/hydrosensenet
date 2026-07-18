"""Core sensor placement algorithms using QR decomposition."""

import warnings

import numpy as np
from scipy.linalg import qr
from typing import Optional, List, Union
import pandas as pd


def _resolve_method(method: str, m: int, n: int, k: int) -> str:
    """Pick the factorization path for an m x n matrix and k sensors.

    The truncated greedy costs O(m*n*k) but runs level-2 BLAS; full
    LAPACK pivoted QR costs O(m*n*min(m,n)) with better constants. The
    crossover measured on Apple/OpenBLAS hardware sits near
    ``5*k == min(m, n)``, and below ~2000 columns the full path is
    always fast enough to prefer.
    """
    if method not in ("auto", "full", "truncated"):
        raise ValueError(
            f"method must be 'auto', 'full', or 'truncated', got {method!r}"
        )
    if method != "auto":
        return method
    if n >= 2000 and 5 * k < min(m, n):
        return "truncated"
    return "full"


def _truncated_pivoted_qr(X: np.ndarray, k: int, copy: bool = True) -> np.ndarray:
    """Greedy k-step column-pivoted QR (Businger-Golub).

    Selects the same pivots as a full pivoted QR truncated to k
    (verified exactly across problem sizes), in O(m*n*k) time and a
    single working copy. Column norms are recomputed from the residual
    at selection time, so no norm-downdating drift accumulates.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (m, n); float32 input stays float32.
    k : int
        Number of pivot columns to select.
    copy : bool, default=True
        Set False only when ``X`` is a temporary this function may
        overwrite.
    """
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    if copy:
        R = np.array(X, dtype=dtype)
    else:
        R = np.ascontiguousarray(X, dtype=dtype)

    n = R.shape[1]
    norms2 = np.einsum("ij,ij->j", R, R)
    pivots = np.empty(k, dtype=np.int64)
    # Rank tolerance relative to the largest column norm, LAPACK-style:
    # once the largest residual falls below it, remaining columns are
    # numerically dependent on the selected ones.
    scale = float(np.sqrt(norms2.max())) if norms2.size else 0.0
    tol = scale * max(R.shape) * np.finfo(R.dtype).eps * 10
    for j in range(k):
        p = int(np.argmax(norms2))
        q = R[:, p]
        nq = np.linalg.norm(q)
        if nq <= tol or not np.isfinite(nq):
            # Residual exhausted: matrix rank < k. Match LAPACK's
            # behavior of still returning k columns by filling with the
            # remaining unselected columns in index order.
            warnings.warn(
                f"Matrix rank ({j}) is smaller than the requested "
                f"{k} sensors; remaining selections are arbitrary.",
                UserWarning,
                stacklevel=3,
            )
            remaining = np.setdiff1d(np.arange(n), pivots[:j], assume_unique=False)
            pivots[j:] = remaining[: k - j]
            return pivots
        pivots[j] = p
        q = q / nq
        coeffs = q @ R
        R -= np.outer(q, coeffs)
        norms2 -= coeffs * coeffs
        np.maximum(norms2, 0.0, out=norms2)
        norms2[pivots[: j + 1]] = -np.inf
    return pivots


def sensor_placement_qr(
    X: np.ndarray,
    n_sensors: int,
    weights: Optional[Union[np.ndarray, List[float]]] = None,
    fixed_indices: Optional[List[int]] = None,
    method: str = "auto",
    verbose: bool = False
) -> np.ndarray:
    """Select sensor locations by (weighted) QR decomposition with column pivoting.

    Columns of ``X`` are candidate locations; pivoting ranks them by how
    much independent information they contribute, and the first
    ``n_sensors`` pivots are returned. Deterministic: the same input
    always yields the same selection.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data matrix of shape ``(n_timesteps, n_locations)``. Must not
        contain NaN/inf (filter beforehand, e.g. with
        :func:`~hydrosensenet.split_timeseries`).
    n_sensors : int
        Number of locations to select.
    weights : array-like of shape (n_locations,), optional
        Per-location importance weights. Columns are scaled by these
        values before pivoting, biasing selection toward high-weight
        locations.
    fixed_indices : list of int, optional
        Column indices of existing sensors to keep. They are placed
        first in the result; remaining sensors are chosen from the
        residual after orthogonalizing against them.
    method : {"auto", "full", "truncated"}, default="auto"
        Factorization path. ``"full"`` runs LAPACK pivoted QR over all
        ``min(m, n)`` elimination steps; ``"truncated"`` runs only the
        ``n_sensors`` greedy steps (O(m*n*k)), selecting identical
        pivots ~10x faster on large problems. ``"auto"`` picks
        ``"truncated"`` when ``n >= 2000`` and ``5 * n_sensors <
        min(m, n)``; float32 input is preserved on the truncated path,
        halving memory.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    np.ndarray
        Selected column indices, length ``n_sensors``. When
        ``fixed_indices`` is given, those indices appear first, in the
        given order.
    """
    # Convert to numpy array if pandas DataFrame
    if hasattr(X, 'values'):
        X = X.values

    if not isinstance(n_sensors, (int, np.integer)) or isinstance(n_sensors, bool) or n_sensors <= 0:
        raise ValueError(f"n_sensors must be a positive integer, got {n_sensors!r}")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN/inf values. Filter them first, e.g. with "
            "split_timeseries(filter_invalid=True) or filter_valid_data()."
        )

    # Convert weights to numpy array if needed
    if weights is not None:
        weights = np.asarray(weights)
        if len(weights) != X.shape[1]:
            raise ValueError(
                f"Weights length ({len(weights)}) must match number of "
                f"locations ({X.shape[1]})"
            )
        # Scale the columns by weights (broadcasting, avoids creating huge diagonal matrix)
        if verbose:
            print(f"      - Applying weights to {X.shape[1]:,} locations...")
        X_w = X * weights[np.newaxis, :]
        x_w_is_temporary = True
    else:
        X_w = X
        x_w_is_temporary = False

    if n_sensors > X.shape[1]:
        raise ValueError(
            f"Requested {n_sensors} sensors but only {X.shape[1]} "
            f"locations available"
        )

    # Handle fixed indices (existing sensors). len() instead of truthiness
    # so numpy arrays work as input.
    if fixed_indices is not None and len(fixed_indices) > 0:
        fixed_indices = [int(i) for i in fixed_indices]

        # Validate fixed indices
        if len(set(fixed_indices)) != len(fixed_indices):
            raise ValueError(f"fixed_indices contains duplicates: {fixed_indices}")
        if min(fixed_indices) < 0:
            raise ValueError(f"fixed_indices must be non-negative, got {min(fixed_indices)}")
        if max(fixed_indices) >= X.shape[1]:
            raise ValueError(
                f"Fixed index {max(fixed_indices)} exceeds number of "
                f"locations ({X.shape[1]})"
            )
        if n_sensors < len(fixed_indices):
            raise ValueError(
                f"n_sensors ({n_sensors}) is smaller than the number of "
                f"fixed_indices ({len(fixed_indices)})"
            )

        if verbose:
            print(f"      - Processing {len(fixed_indices)} fixed sensor locations...")

        # Partition matrix into fixed and free columns
        if X_w.dtype not in (np.float32, np.float64):
            X_w = np.asarray(X_w, dtype=np.float64)
        A_F = X_w[:, fixed_indices]
        free_mask = np.ones(X.shape[1], dtype=bool)
        free_mask[fixed_indices] = False
        free_indices = np.where(free_mask)[0]
        A_R = X_w[:, free_indices]

        # QR on fixed columns
        if verbose:
            print(f"      - QR decomposition on fixed sensors...")
        Q_F, R_F = np.linalg.qr(A_F)

        # Orthogonalize free columns with respect to Q_F
        if verbose:
            print(f"      - Orthogonalizing {len(free_indices):,} free locations...")
        projection = Q_F @ (Q_F.T @ A_R)
        A_R_prime = A_R - projection

        # Pivoted QR on the orthogonalized free columns
        k_free = n_sensors - len(fixed_indices)
        run = _resolve_method(method, X.shape[0], len(free_indices), max(k_free, 1))
        if run == "truncated":
            if verbose:
                print(f"      - Running truncated ({k_free}-step) pivoted QR on free locations...")
            # A_R_prime is a fresh temporary, safe to overwrite
            pivots_R = _truncated_pivoted_qr(A_R_prime, k_free, copy=False)
        else:
            if verbose:
                print(f"      - Running pivoted QR on free locations...")
            # Use mode='r' to skip computing Q
            R_R, pivots_R = qr(A_R_prime, pivoting=True, mode='r')

        # Combine fixed and pivoted columns
        pivots = np.concatenate(
            [np.asarray(fixed_indices), free_indices[np.asarray(pivots_R)]]
        )
    else:
        run = _resolve_method(method, X.shape[0], X.shape[1], n_sensors)
        if run == "truncated":
            # Greedy k-step pivoting: same selection as the full
            # factorization at O(m*n*k) cost; float32 input stays float32.
            if verbose:
                print(f"      - Running truncated ({n_sensors}-step) pivoted QR...")
            pivots = _truncated_pivoted_qr(X_w, n_sensors, copy=not x_w_is_temporary)
        else:
            # Standard pivoted QR on weighted matrix
            if verbose:
                print(f"      - Running pivoted QR decomposition...")
            # Fortran order and float64 for LAPACK; mode='r' skips computing Q
            X_full = np.asarray(X_w, dtype=np.float64, order='F')
            R, pivots = qr(X_full, pivoting=True, mode='r')
            pivots = np.array(pivots)

    # Select the first n_sensors columns from the permutation
    if n_sensors > len(pivots):
        raise ValueError(
            f"Requested {n_sensors} sensors but only {len(pivots)} "
            f"locations available"
        )

    selected_indices = pivots[:n_sensors]

    return selected_indices


def qr_pivot_selection(
    X_train: np.ndarray,
    region_assignments: Union[pd.DataFrame, np.ndarray],
    sensors_per_region: dict,
    region_column: str = "region_name",
    verbose: bool = False
) -> tuple:
    """Select sensors independently within each spatial region (per-region quotas).

    Runs pivoted QR separately on each region's columns, taking the
    region's quota from ``sensors_per_region``.

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Data matrix of shape ``(n_timesteps, n_locations)``.
    region_assignments : pd.DataFrame or np.ndarray
        Region label per location. As a DataFrame it must contain
        ``region_column`` and a ``col_pos`` column giving each row's
        column position in ``X_train``; as an array, positions are
        assigned in order.
    sensors_per_region : dict
        Mapping of region name to number of sensors. Regions not listed
        get zero sensors; quotas are capped at the region's size.
    region_column : str, default="region_name"
        Name of the region label column.
    verbose : bool, default=False
        Print a summary of the selection.

    Returns
    -------
    selected_locations : pd.DataFrame
        Rows of ``region_assignments`` for the selected locations.
    selected_sensor_indices : list of int
        Selected column positions into ``X_train``.
    """
    # Convert to numpy array if pandas DataFrame
    if hasattr(X_train, 'values'):
        X_train = X_train.values

    # Handle array input
    if isinstance(region_assignments, np.ndarray):
        region_assignments = pd.DataFrame({
            region_column: region_assignments,
            'col_pos': np.arange(len(region_assignments))
        })

    if not isinstance(region_assignments, pd.DataFrame):
        raise TypeError("region_assignments must be DataFrame or array")

    # Ensure col_pos column exists
    if 'col_pos' not in region_assignments.columns:
        raise ValueError("region_assignments must have 'col_pos' column")

    selected_rows = []
    selected_sensor_indices = []

    # Process each region
    for region_name, group in region_assignments.groupby(region_column):
        # Get number of sensors for this region
        k = sensors_per_region.get(region_name, 0)
        if k == 0:
            continue

        # Get column positions for this region
        positions = group['col_pos'].to_numpy()
        if len(positions) == 0:
            continue

        # Limit to available positions
        k = min(k, len(positions))

        # Perform QR decomposition on this region's data
        # Use mode='r' to skip computing Q
        _, pivots = qr(X_train[:, positions], pivoting=True, mode='r')

        # Select top k locations
        chosen_positions = positions[pivots[:k]]
        selected_sensor_indices.extend(chosen_positions)

        # Store selected locations info
        selected = (group.set_index('col_pos').loc[chosen_positions].reset_index())
        selected_rows.append(selected)

    # Combine all selected locations
    if selected_rows:
        selected_locations = pd.concat(selected_rows, ignore_index=True)
    else:
        selected_locations = pd.DataFrame()

    if verbose:
        print(f"\nSelected {len(selected_sensor_indices)} optimal sensor locations")

    return selected_locations, selected_sensor_indices
