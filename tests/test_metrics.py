"""Correctness tests for evaluation metrics and field reconstruction."""
import numpy as np
import pytest

from hydrosensenet import (
    calculate_performance_metrics,
    reconstruction_evaluation,
    sensor_placement_qr,
)
from hydrosensenet.core import reconstruction_error


def test_perfect_prediction_gives_unit_scores():
    rng = np.random.default_rng(0)
    true = rng.uniform(1, 10, size=(50, 4))
    r2, nse, nnse = calculate_performance_metrics(true, true.copy())
    np.testing.assert_allclose(r2, 1.0)
    np.testing.assert_allclose(nse, 1.0)
    np.testing.assert_allclose(nnse, 1.0)


def test_known_nse_value():
    # true column: [0, 2] -> mean 1, ss_tot = 2
    # pred column: [1, 1] -> ss_res = 2 -> NSE = 0, NNSE = 1/(2-0) = 0.5
    true = np.array([[0.0], [2.0]])
    pred = np.array([[1.0], [1.0]])
    r2, nse, nnse = calculate_performance_metrics(true, pred)
    np.testing.assert_allclose(nse, [0.0])
    np.testing.assert_allclose(r2, [0.0])
    np.testing.assert_allclose(nnse, [0.5])


def test_constant_true_column_gives_nan():
    true = np.full((10, 1), 3.0)
    pred = np.full((10, 1), 2.0)
    _, nse, _ = calculate_performance_metrics(true, pred)
    assert np.isnan(nse[0])


def test_reconstruction_error_known_values():
    true = np.array([[0.0, 0.0], [3.0, 4.0]])
    pred = true + 1.0

    np.testing.assert_allclose(reconstruction_error(true, pred, "rmse"), [1.0, 1.0])
    np.testing.assert_allclose(reconstruction_error(true, pred, "mae"), [1.0, 1.0])
    # ||ones(2x2)||_F = 2, ||true||_F = 5
    np.testing.assert_allclose(reconstruction_error(true, pred, "relative"), 0.4)
    np.testing.assert_allclose(reconstruction_error(true, pred, "frobenius"), 2.0)


def test_reconstruction_error_unknown_metric_raises():
    true = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Unknown metric"):
        reconstruction_error(true, true, "nope")


def test_reconstruction_recovers_low_rank_field():
    """A rank-3 field must be recovered near-exactly from 3 QR-chosen sensors."""
    rng = np.random.default_rng(1)
    U = rng.uniform(0.5, 1.5, size=(200, 3))
    V = rng.uniform(0.5, 2.0, size=(3, 20))
    X = U @ V  # positive, rank 3

    X_train, X_test = X[:140], X[140:]
    sensors = sensor_placement_qr(X_train, n_sensors=3)

    results = reconstruction_evaluation(X_train, X_test, sensors, n_sensors=3)

    assert results["relative_error"] < 1e-6
    np.testing.assert_allclose(results["nse"], 1.0, atol=1e-6)
    assert results["X_test_reconstructed"].shape == X_test.shape
    assert len(results["selected_sensors"]) == 3
    assert len(results["non_selected_sensors"]) == 17


def test_reconstruction_evaluation_partitions_sensors():
    rng = np.random.default_rng(2)
    X = rng.uniform(1, 5, size=(100, 10))
    sensors = np.array([1, 4, 7, 9])

    results = reconstruction_evaluation(X[:70], X[70:], sensors, n_sensors=2)

    np.testing.assert_array_equal(results["selected_sensors"], [1, 4])
    assert set(results["non_selected_sensors"]) == set(range(10)) - {1, 4}


def test_reconstruction_stable_with_near_collinear_sensors():
    """Regression: direct lstsq must not square the condition number."""
    rng = np.random.default_rng(0)
    n_t, eps = 300, 1e-7
    base = rng.uniform(1, 2, (n_t, 1))
    S = np.hstack([
        base,
        base * (1 + eps * rng.standard_normal((n_t, 1))),
        rng.uniform(1, 2, (n_t, 1)),
    ])
    X = np.hstack([S, S @ rng.uniform(0.5, 1.5, (3, 5))])

    results = reconstruction_evaluation(X[:200], X[200:], np.array([0, 1, 2]), 3)

    # The Gram-matrix (normal equations) formulation loses ~half the
    # significant digits here; the direct solver stays near machine precision.
    assert results["relative_error"] < 1e-10


def test_reconstruction_preserves_negative_values():
    """Regression: no positivity clamp for data that is not non-negative."""
    rng = np.random.default_rng(4)
    U = rng.standard_normal((200, 2))
    V = rng.standard_normal((2, 10))
    X = U @ V  # anomaly-like field with plenty of negative values
    assert (X < 0).any()

    sensors = sensor_placement_qr(X[:140], n_sensors=2)
    results = reconstruction_evaluation(X[:140], X[140:], sensors, n_sensors=2)

    # Exact rank-2 recovery is only possible if negatives are not clamped
    assert results["relative_error"] < 1e-8
    assert (results["X_test_reconstructed"] < 0).any()
