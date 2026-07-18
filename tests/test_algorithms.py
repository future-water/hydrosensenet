"""Correctness tests for the QR-based sensor placement algorithms."""
import numpy as np
import pandas as pd
import pytest

from hydrosensenet import sensor_placement_qr, qr_pivot_selection


def _signal_matrix(n_timesteps=200, seed=0):
    """Matrix whose first 3 columns carry independent strong signals.

    Columns 3-7 are near-duplicates of those signals at 1/1000 the
    amplitude, so a correct QR pivoting must pick columns 0-2 first.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    signals = [np.sin(t), np.cos(t), np.sin(2 * t)]

    X = np.zeros((n_timesteps, 8))
    for j, s in enumerate(signals):
        X[:, j] = 10.0 * s
    for j, s in zip(range(3, 8), [signals[0], signals[1], signals[2], signals[0], signals[1]]):
        X[:, j] = 0.01 * s + rng.normal(scale=1e-3, size=n_timesteps)
    return X


def test_qr_selects_informative_columns():
    X = _signal_matrix()
    selected = sensor_placement_qr(X, n_sensors=3)
    assert set(selected) == {0, 1, 2}


def test_qr_is_deterministic():
    X = _signal_matrix()
    first = sensor_placement_qr(X, n_sensors=5)
    second = sensor_placement_qr(X, n_sensors=5)
    np.testing.assert_array_equal(first, second)


def test_selected_indices_are_unique_and_in_range():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 15))
    selected = sensor_placement_qr(X, n_sensors=10)
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert all(0 <= i < 15 for i in selected)


def test_weights_bias_selection():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 8))
    weights = np.ones(8)
    weights[4] = 100.0
    selected = sensor_placement_qr(X, n_sensors=1, weights=weights)
    assert selected[0] == 4


def test_weights_length_mismatch_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="Weights length"):
        sensor_placement_qr(X, n_sensors=2, weights=np.ones(5))


def test_too_many_sensors_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="only"):
        sensor_placement_qr(X, n_sensors=7)


def test_fixed_indices_come_first():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 10))
    selected = sensor_placement_qr(X, n_sensors=4, fixed_indices=[2, 5])
    assert list(selected[:2]) == [2, 5]
    assert len(set(selected)) == 4


def test_fixed_index_out_of_range_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="Fixed index"):
        sensor_placement_qr(X, n_sensors=3, fixed_indices=[6])


def test_accepts_dataframe_input():
    X = _signal_matrix()
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(X.shape[1])])
    selected = sensor_placement_qr(df, n_sensors=3)
    assert set(selected) == {0, 1, 2}


def test_qr_pivot_selection_respects_regions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6))
    assignments = pd.DataFrame({
        "region_name": ["A", "A", "A", "B", "B", "B"],
        "col_pos": np.arange(6),
    })

    locations, indices = qr_pivot_selection(
        X, assignments, sensors_per_region={"A": 2, "B": 1}
    )

    assert len(indices) == 3
    assert len(locations) == 3
    region_a = [i for i in indices if i in {0, 1, 2}]
    region_b = [i for i in indices if i in {3, 4, 5}]
    assert len(region_a) == 2
    assert len(region_b) == 1


def test_qr_pivot_selection_skips_unlisted_regions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6))
    assignments = pd.DataFrame({
        "region_name": ["A", "A", "A", "B", "B", "B"],
        "col_pos": np.arange(6),
    })

    _, indices = qr_pivot_selection(X, assignments, sensors_per_region={"A": 2})

    assert len(indices) == 2
    assert set(indices) <= {0, 1, 2}


def test_qr_pivot_selection_requires_col_pos():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    assignments = pd.DataFrame({"region_name": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="col_pos"):
        qr_pivot_selection(X, assignments, sensors_per_region={"A": 1})


def test_fixed_indices_accepts_numpy_array():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 10))
    selected = sensor_placement_qr(X, n_sensors=4, fixed_indices=np.array([2, 5]))
    assert list(selected[:2]) == [2, 5]


def test_fixed_indices_duplicates_raise():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="duplicate"):
        sensor_placement_qr(X, n_sensors=3, fixed_indices=[2, 2])


def test_fixed_indices_negative_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="non-negative"):
        sensor_placement_qr(X, n_sensors=3, fixed_indices=[-1])


def test_n_sensors_smaller_than_fixed_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    with pytest.raises(ValueError, match="smaller"):
        sensor_placement_qr(X, n_sensors=1, fixed_indices=[0, 1])


def test_invalid_n_sensors_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            sensor_placement_qr(X, n_sensors=bad)


def test_nonfinite_input_raises_clear_error():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sensor_placement_qr(X, n_sensors=2)


def _low_rank_noisy(m, n, rank=16, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.5, 1.5, (m, rank))
    V = rng.uniform(0.5, 2.0, (rank, n))
    return U @ V + 0.01 * np.abs(rng.standard_normal((m, n)))


def test_truncated_matches_full_selection():
    X = _low_rank_noisy(300, 600)
    full = sensor_placement_qr(X, n_sensors=20, method="full")
    trunc = sensor_placement_qr(X, n_sensors=20, method="truncated")
    np.testing.assert_array_equal(full, trunc)


def test_truncated_matches_full_with_weights():
    X = _low_rank_noisy(300, 400, seed=1)
    rng = np.random.default_rng(2)
    weights = rng.uniform(0.5, 3.0, 400)
    full = sensor_placement_qr(X, n_sensors=10, weights=weights, method="full")
    trunc = sensor_placement_qr(X, n_sensors=10, weights=weights, method="truncated")
    np.testing.assert_array_equal(full, trunc)


def test_truncated_matches_full_with_fixed_indices():
    X = _low_rank_noisy(300, 400, seed=3)
    full = sensor_placement_qr(X, n_sensors=10, fixed_indices=[3, 7], method="full")
    trunc = sensor_placement_qr(X, n_sensors=10, fixed_indices=[3, 7], method="truncated")
    np.testing.assert_array_equal(full, trunc)


def test_auto_method_matches_full_on_large_problem():
    # n >= 2000 and 5*k < min(m, n) -> auto dispatches to truncated
    X = _low_rank_noisy(150, 2500, seed=4)
    auto = sensor_placement_qr(X, n_sensors=5, method="auto")
    full = sensor_placement_qr(X, n_sensors=5, method="full")
    np.testing.assert_array_equal(auto, full)


def test_invalid_method_raises():
    X = _low_rank_noisy(50, 60)
    with pytest.raises(ValueError, match="method"):
        sensor_placement_qr(X, n_sensors=2, method="fast")


def test_truncated_float32_input():
    X = _low_rank_noisy(200, 2500, seed=5).astype(np.float32)
    selected = sensor_placement_qr(X, n_sensors=5, method="truncated")
    full = sensor_placement_qr(X.astype(np.float64), n_sensors=5, method="full")
    np.testing.assert_array_equal(selected, full)


def test_truncated_rank_deficient_fills_and_warns():
    rng = np.random.default_rng(6)
    U = rng.standard_normal((50, 2))
    V = rng.standard_normal((2, 10))
    X = U @ V  # exact rank 2
    with pytest.warns(UserWarning, match="rank"):
        selected = sensor_placement_qr(X, n_sensors=5, method="truncated")
    assert len(selected) == 5
    assert len(set(selected)) == 5
