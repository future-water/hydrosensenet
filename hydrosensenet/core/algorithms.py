"""Core sensor placement algorithms using QR decomposition."""

import numpy as np
from scipy.linalg import qr
from typing import Optional, List, Union
import pandas as pd


def sensor_placement_qr(
    X: np.ndarray,
    n_sensors: int,
    weights: Optional[Union[np.ndarray, List[float]]] = None,
    fixed_indices: Optional[List[int]] = None,
    verbose: bool = False
) -> np.ndarray:
    """Perform sensor placement using weighted QR decomposition."""
    # Convert to numpy array if pandas DataFrame
    if hasattr(X, 'values'):
        X = X.values

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
    else:
        X_w = X

    # Use Fortran order for better LAPACK performance
    if verbose:
        print(f"      - Converting to Fortran order for LAPACK...")
    X_w = np.asarray(X_w, dtype=np.float64, order='F')

    # Handle fixed indices (existing sensors)
    if fixed_indices:
        fixed_indices = list(fixed_indices)

        # Validate fixed indices
        if max(fixed_indices) >= X.shape[1]:
            raise ValueError(
                f"Fixed index {max(fixed_indices)} exceeds number of "
                f"locations ({X.shape[1]})"
            )

        if verbose:
            print(f"      - Processing {len(fixed_indices)} fixed sensor locations...")

        # Partition matrix into fixed and free columns
        A_F = X_w[:, fixed_indices]
        free_indices = [i for i in range(X.shape[1]) if i not in fixed_indices]
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

        # Pivoted QR on orthogonalized free columns
        if verbose:
            print(f"      - Running pivoted QR on free locations...")
        # Use mode='r' to skip computing Q
        R_R, pivots_R = qr(A_R_prime, pivoting=True, mode='r')

        # Combine fixed and pivoted columns
        pivots = np.array(fixed_indices + [free_indices[i] for i in pivots_R])
    else:
        # Standard pivoted QR on weighted matrix
        if verbose:
            print(f"      - Running pivoted QR decomposition...")
        # Use mode='r' to skip computing Q
        R, pivots = qr(X_w, pivoting=True, mode='r')
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
    region_column: str = "region_name"
) -> tuple:
    """Perform QR-pivot sensor selection within each spatial region."""
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
        selected = group.loc[
            group['col_pos'].isin(chosen_positions)
        ].copy()
        selected_rows.append(selected)

    # Combine all selected locations
    if selected_rows:
        selected_locations = pd.concat(selected_rows, ignore_index=True)
    else:
        selected_locations = pd.DataFrame()

    print(f"\nSelected {len(selected_sensor_indices)} optimal sensor locations")

    return selected_locations, selected_sensor_indices
