# Evaluation and metrics

Selecting sensor locations is only half the problem: you also need evidence that the
chosen network can reproduce conditions at the *ungauged* locations. hydrosensenet
evaluates a network by holding out the end of the record, learning a linear
reconstruction map on the training period, and scoring it against the held-out test
period. This page walks through that protocol with the low-level functions, then
shows how the same numbers surface through the high-level API. If you are new here,
start with the [quickstart](../quickstart.md) and the [network design guide](design.md).

## Temporal train/test splitting

{func}`hydrosensenet.split_timeseries` performs a *temporal* split: the first
`train_frac` of the timesteps become training data and the remainder becomes test data.
The split is never random — shuffling daily streamflow would leak autocorrelated
information from the test period into training and inflate every score. With
`filter_invalid=True` (the default), any location (column) containing NaN or inf
*within the training window* is dropped from both splits, since the least-squares fit
cannot use it; pass `return_mapping=True` to see which columns survived. Note that for
`xarray.Dataset` inputs only the temporal split is applied, without column filtering.

```python
import numpy as np
import geopandas as gpd

from hydrosensenet import (
    SensorNetworkDesigner,
    split_timeseries,
    sensor_placement_qr,
    reconstruction_evaluation,
)

rng = np.random.default_rng(42)
n_time, n_locations, rank = 730, 60, 5

# Strictly positive rank-5 "streamflow" field: 5 latent hydrologic modes
# shared across 60 locations (two years of daily values)
modes = rng.random((n_time, rank)) + 0.1
loadings = rng.random((rank, n_locations)) + 0.1
field = modes @ loadings

# Inject a gap at one location to show the filtering behavior
field[10, 3] = np.nan

X_train, X_test, mapping = split_timeseries(
    field, train_frac=0.7, filter_invalid=True, return_mapping=True
)
print(X_train.shape, X_test.shape, mapping["n_removed"])
# (510, 59) (220, 59) 1  -- the gappy column was dropped from both splits
```

## How reconstruction works

Given sensors at a column subset *s*, {func}`hydrosensenet.reconstruction_evaluation`
learns a linear map from the sensor readings to the full field on the training data,
then applies it to the test-period sensor readings:

```{math}
\hat{X}_{\text{test}} = X_{\text{test},s}\, B^{\star},
\qquad
B^{\star} = \arg\min_{B}\ \lVert X_{\text{train},s}\, B - X_{\text{train}} \rVert_F^{2}
```

Internally the least-squares operator is solved directly (SVD-based `lstsq` on
the tall sensor matrix), which stays numerically stable even for
nearly-collinear sensor subsets. When the input data is entirely non-negative
(as streamflow is), the reconstruction is floored at `0`; variables that can
legitimately go negative, such as anomalies, are left unclamped.

The returned dictionary contains both the reconstruction and its scores:

`X_test_reconstructed`, `X_test_selected`
: Reconstructed test-period field `(n_test, n_locations)` and the raw sensor readings.

`selected_sensors`, `non_selected_sensors`
: Column indices used as sensors (the first `n_sensors` of your ranking) and the rest.

`rmse`, `nse`, `nnse`
: Per-location arrays scored over the test period.

`relative_error`
: A single scalar, {math}`\lVert \hat{X} - X \rVert_F / \lVert X \rVert_F`, for the
  whole field.

Because our synthetic field has rank 5, any 8 well-placed sensors span it exactly and
recovery is near-perfect — a useful sanity check for your own pipeline:

```python
sensors = sensor_placement_qr(X_train, n_sensors=8)
results = reconstruction_evaluation(X_train, X_test, sensors, n_sensors=8)
print(f"relative error: {results['relative_error']:.2e}")   # ~1e-14
print(f"median NNSE:    {np.nanmedian(results['nnse']):.4f}")  # 1.0000
```

## Metric definitions

For each location, NSE (Nash–Sutcliffe efficiency) over the test period is
{math}`1 - \mathrm{SS}_{\mathrm{res}} / \mathrm{SS}_{\mathrm{tot}}`: 1 is a perfect
reconstruction, 0 means "no better than predicting the test-period mean", and it is
unbounded below. One honest caveat: the `r_squared` returned by
{func}`hydrosensenet.calculate_performance_metrics` is computed with this exact same
formula, so it is identical to NSE — the designer simply omits it.

NNSE (normalized NSE) is {math}`1 / (2 - \mathrm{NSE})`, which maps
{math}`(-\infty, 1]` onto {math}`(0, 1]` while preserving ranking (NSE 1 → NNSE 1;
NSE 0 → NNSE 0.5). This bounded range is much friendlier for mapping and aggregation:
a single catastrophically bad reach cannot drag a basin-wide mean toward negative
infinity or blow out a map's color scale. Both metrics are NaN wherever the
test-period series is constant ({math}`\mathrm{SS}_{\mathrm{tot}} = 0`, e.g. a reach
that stays dry) — use `np.nanmedian` and friends when aggregating.

## Choosing the network size

QR pivot rankings are *nested*: the first *k* entries of a 20-sensor ranking are
exactly what a *k*-sensor call would return. So one call to
{func}`hydrosensenet.sensor_placement_qr` supports a whole sweep of network sizes, and
`reconstruction_evaluation`'s `n_sensors` argument does the truncation. Sweep the count
on data with realistic noise and look for the knee where median NNSE plateaus —
sensors beyond that point mostly buy redundancy:

```python
good = mapping["good_cols"]
noisy = field[:, good] + rng.normal(0.0, 0.05, size=(n_time, len(good)))
noisy = np.clip(noisy, 1e-6, None)  # keep the synthetic field positive
Xn_train, Xn_test = split_timeseries(noisy, train_frac=0.7)

ranking = sensor_placement_qr(Xn_train, n_sensors=20)
print(f"{'n_sensors':>9} {'median NNSE':>12} {'rel. error':>11}")
for k in (2, 4, 6, 8, 12, 20):
    r = reconstruction_evaluation(Xn_train, Xn_test, ranking, n_sensors=k)
    print(f"{k:>9} {np.nanmedian(r['nnse']):>12.3f} {r['relative_error']:>11.4f}")
# Scores climb steeply up to the field's effective rank (~5), then flatten.
```

The same sweep on the bundled real sample basin (see the
[quickstart](../quickstart.md)) shows what to expect outside of clean
synthetic fields — steep early gains, then diminishing returns:

```{figure} ../_static/sample_size_sweep.png
:alt: Two panels versus number of sensors for the 368-reach sample basin; median per-reach NNSE rises from 0.51 at five sensors to essentially 1.0 at one hundred, while flow-weighted relative error falls more than an order of magnitude on a log scale.

Network-size sweep on the lower Colorado sample (368 reaches, trained
on 2020, evaluated on 2021). Median NNSE climbs from 0.51 (5 sensors)
through 0.71 (20) to ~1.0 (100), and the flow-weighted relative error
collapses by more than an order of magnitude — steep early gains,
then diminishing returns past the knee.
```

## Per-location results from the high-level API

{meth}`~hydrosensenet.SensorNetworkDesigner.design_network` with `evaluate=True` (the
default) runs this entire protocol for you, splitting with its `train_frac` argument.
{meth}`~hydrosensenet.NetworkDesignResult.get_dataframe` returns one row per *selected*
sensor (`sensor_rank`, `location_index`, `location_label`, coordinates, `nse`, `nnse`),
while `result.performance_metrics["nnse"]` holds the full per-location array —
including ungauged reaches — and `performance_metrics["eval_results"]` keeps the raw
dictionary from `reconstruction_evaluation`.

```python
lons = rng.uniform(-112.0, -109.0, len(good))
lats = rng.uniform(39.0, 41.5, len(good))
gauges = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")

designer = SensorNetworkDesigner(noisy, gauges)
result = designer.design_network(n_sensors=8, verbose=False)

print(result.get_dataframe()[["sensor_rank", "location_index", "nnse"]].head())
nnse_all = result.performance_metrics["nnse"]
print(f"locations with NNSE > 0.9: {np.mean(nnse_all > 0.9):.0%}")
```

One caution: the designer's internal split always uses `filter_invalid=True`, so
dropped columns shift indices relative to your locations GeoDataFrame. Remove invalid
locations with {func}`hydrosensenet.filter_valid_data` (see the [data guide](data.md))
*before* constructing the designer so indices and geometries stay aligned.
