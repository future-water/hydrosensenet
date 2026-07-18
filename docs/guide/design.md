# Designing a network

This page covers the placement engine: how the algorithm chooses locations, the
options on {class}`hydrosensenet.SensorNetworkDesigner`, and the lower-level
functions. New to the package? Start with the [quickstart](../quickstart.md).

## How QR column pivoting picks sensors

Arrange your historical record as a matrix `X` with one row per timestep and one
column per candidate location, so each column is that location's hydrograph.
Column-pivoted QR factorizes `X` while greedily reordering its columns: at every
step it selects the column with the largest norm *after projecting out
everything already selected* — the location carrying the most variance that the
chosen sensors cannot already explain. The first `k` pivots are therefore both
maximally informative and mutually non-redundant: a gauge whose record closely
mirrors an already-selected gauge is skipped, even if it is individually
high-variance. The selected columns form a well-conditioned basis for
reconstructing the full flow field from `k` point measurements — exactly what
the evaluation step measures.

## Design options

{meth}`~hydrosensenet.SensorNetworkDesigner.design_network` has three knobs you
will touch most often:

- `n_sensors` — total network size (including any locked-in gauges, see below).
- `evaluate` / `train_frac` — with `evaluate=True` (default) the record is
  split chronologically (`train_frac=0.7`: first 70% train, last 30% test),
  placement runs on the training window only, and the test window is
  reconstructed from the selected sensors to give NSE/NNSE and relative-error
  metrics. `evaluate=False` places on the full record; no metrics are computed.
- `verbose` — prints a step-by-step progress log; set `False` for scripts.

Columns containing NaN or inf in the training window are excluded from
placement (a warning reports how many). Everything in the result — selected
indices, labels, per-location metric arrays — still refers to the original
column order, and metric arrays carry `NaN` at excluded locations. An
`existing_sensors` entry pointing at an excluded column raises a `ValueError`.

```python
import numpy as np
import geopandas as gpd
from hydrosensenet import SensorNetworkDesigner

rng = np.random.default_rng(42)
n_times, n_locs = 400, 30

# Synthetic streamflow: shared basin-scale modes plus local noise
# (the low-rank structure that real river networks exhibit).
modes = rng.standard_normal((n_times, 4))
flows = modes @ rng.standard_normal((4, n_locs))
flows += 0.1 * rng.standard_normal((n_times, n_locs))

locations = gpd.GeoDataFrame(
    {"gauge_id": [f"G{i:03d}" for i in range(n_locs)]},
    geometry=gpd.points_from_xy(
        rng.uniform(-112.0, -109.0, n_locs), rng.uniform(38.0, 41.0, n_locs)
    ),
    crs="EPSG:4326",
)

designer = SensorNetworkDesigner(flows, locations, list(locations["gauge_id"]))
result = designer.design_network(
    n_sensors=8, train_frac=0.7, evaluate=True, verbose=False
)
result.print_summary()
```

## Locking in existing gauges

If part of the network already exists, pass `existing_sensors=[...]` — integer
indices into the column order of your streamflow matrix. Those columns are kept
unconditionally; the remaining slots are filled by pivoted QR on the other
columns after orthogonalizing them against the fixed ones, so new sensors only
chase information the existing gauges do not provide. The result ordering is
guaranteed: fixed indices come first in `result.selected_indices`, in the order
you passed them, followed by the QR-ranked additions. If
`len(existing_sensors) == n_sensors`, QR is skipped entirely and the call simply
evaluates the network you specified — a convenient way to benchmark a current
gauge configuration.

```python
locked = [2, 17]
result_locked = designer.design_network(
    n_sensors=8, existing_sensors=locked, evaluate=False, verbose=False
)
assert list(result_locked.selected_indices[:2]) == locked
```

## Weighting the design

`weights=` biases selection toward locations you care about. An ndarray with one
entry per column scales each hydrograph before the factorization, so a weight of
3 triples a column's norm and makes it far more likely to be pivoted early; a
weight of 0 removes a location from contention.

```python
weights = np.ones(n_locs)
weights[:10] = 3.0  # prioritize the first ten locations
result_weighted = designer.design_network(
    n_sensors=8, weights=weights, evaluate=False, verbose=False
)
```

Alternatively, pass a file path plus `weight_column`, and the weights are
derived by spatially joining your locations to that dataset via
{func}`hydrosensenet.calculate_spatial_weights`. The following display-only
example needs a downloaded FEMA National Risk Index file; see
[risk-informed design](risk.md) for the full workflow.

```{code-block} python
result = designer.design_network(
    n_sensors=8,
    weights="NRI_GDB_Counties.gpkg",  # any file geopandas can read
    weight_column="RISK_SCORE",
)
```

## Using the modular API directly

When you want the raw index selection without the designer's splitting,
evaluation, and result packaging, call
{func}`hydrosensenet.sensor_placement_qr` on your own training matrix. It has
the same `weights` and `fixed_indices` semantics and returns the selected
column indices, fixed ones first.

```python
from hydrosensenet import sensor_placement_qr

X_train = flows[: int(0.7 * n_times)]
idx = sensor_placement_qr(X_train, n_sensors=8, weights=weights, fixed_indices=[2, 17])
print(idx)
```

For per-region quotas — for example, guaranteeing coverage in each HUC or state
— use {func}`hydrosensenet.qr_pivot_selection`. It runs an independent
(unweighted) pivoted QR inside each region. `region_assignments` is a DataFrame
with a `region_name` column (configurable via `region_column`) and a `col_pos`
column giving each location's column index; `sensors_per_region` maps region
names to quotas (capped at region size; regions absent from the dict get none).
It returns the selected rows of `region_assignments` plus the flat list of
selected column indices.

```python
import pandas as pd
from hydrosensenet import qr_pivot_selection

regions = pd.DataFrame({
    "region_name": np.where(locations.geometry.x < -110.5, "west", "east"),
    "col_pos": np.arange(n_locs),
})
selected_df, selected_idx = qr_pivot_selection(
    X_train, regions, sensors_per_region={"west": 3, "east": 5}
)
print(selected_df[["region_name", "col_pos"]])
```

## Determinism

Placement is fully deterministic: pivoted QR involves no random seed, so the
same matrix, weights, and fixed indices always yield the same selection. The
seed above only generates the synthetic data. Selected indices are always
positions in the column order you supplied, even when the NaN/inf filter
excludes some columns from consideration.

## Scalability

{func}`hydrosensenet.sensor_placement_qr` picks between two factorization
paths via its `method` parameter (default `"auto"`). For large basins it runs
a truncated greedy pivoted QR that performs only the `n_sensors` elimination
steps — O(m·n·k) instead of the full factorization's O(m·n·min(m, n)) — and
selects **identical** sensors. On an Apple M1 Pro with a 3,000 × 64,954
matrix (HUC2-scale) and 50 sensors, the full path takes ~168 s while the
truncated path takes ~14 s. Passing a `float32` matrix halves memory on the
truncated path. Force a specific path with `method="full"` or
`method="truncated"`; `benchmarks/bench_placement.py` in the repository
reproduces the comparison on your hardware.
